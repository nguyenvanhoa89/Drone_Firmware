use std::f32;
use std::thread;
use std::sync::Mutex;
use std::sync::atomic::{ATOMIC_BOOL_INIT, AtomicBool, Ordering};

use mavlink;
use mavlink::common::*;

#[derive(Debug, Clone, Serialize)]
pub struct Telemetry {
    pub position: [f32; 4],
    pub heading: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Location {
    pub x: f32,
    pub y: f32,
    pub alt: f32,
    pub yaw: f32,
}

#[derive(Copy, Clone, Default)]
pub struct SharedData {
    pub position: [f32; 4],
    pub velocity: [f32; 3],
    pub heading: f32,
    pub next_target: Option<[f32; 4]>
}

lazy_static! {
    static ref MAVLINK_DATA: Mutex<SharedData> = Mutex::new(SharedData::default());
}

pub fn get_telemetry() -> Telemetry {
    let mavlink_data = MAVLINK_DATA.lock().unwrap();

    Telemetry {
        position: mavlink_data.position,
        heading: mavlink_data.heading,
    }
}

pub fn do_reposition(target: Location) {
    MAVLINK_DATA.lock().unwrap().next_target = Some([target.x, target.y, target.alt,  target.yaw]);
}

static STOPPED: AtomicBool = ATOMIC_BOOL_INIT;

pub struct MavlinkHandle {}

impl Drop for MavlinkHandle {
    fn drop(&mut self) {
        STOPPED.store(true, Ordering::Relaxed);
    }
}

impl MavlinkHandle {
    pub fn new(addr: String) -> MavlinkHandle {
        STOPPED.store(false, Ordering::Relaxed);
        *MAVLINK_DATA.lock().unwrap() = SharedData::default();
        thread::spawn(move || mavlink_background_process(addr));
        MavlinkHandle { }
    }
}

const EARTH_RADIUS_METERS: f64 = 6371e3;

#[derive(Default)]
struct GpsBase {
    base: Option<Coordinate>,
}

impl GpsBase {
    fn next(&mut self, lon: i32, lat: i32) -> Option<[f32; 2]> {
        let new = Coordinate { lat: lat as f64 / 1e7, lon: lon as f64 / 1e7 };

        match self.base {
            Some(base) => {
            	let lat_average = (new.lat + base.lat).to_radians()/2.0;
				let lat_average_cos = lat_average.cos();
                let x_offset = (new.lon - base.lon).to_radians() * EARTH_RADIUS_METERS * lat_average_cos;
                let y_offset = (new.lat - base.lat) * 110540.0 ; 
                Some([x_offset as f32, y_offset as f32])
            },
            None => {
                self.base = Some(new);
                None
            }
        }
    }

    fn invert(&self, x: f32, y: f32) -> Coordinate {
        let base = self.base.expect("Tried to invert offset without base");
		let rad_rate = f32::consts::PI/180.0 ;
		let lat = rad_rate * (base.lat as f32) as f32;
		let lat_cos = lat.cos();
		let delta_longitude = x /(111320.0 * lat_cos as f32)  ; // 111320 = EARTH_RADIUS_METERS * pi /180
		let delta_latitude = y/110540.0                    ; // result in degrees long/lat
		// The difference between the constants 110540 and 111320 is due to the earth's oblateness (polar and equatorial circumferences are different).
		// Source: http://stackoverflow.com/questions/2187657/calculate-second-point-knowing-the-starting-point-and-distance
        Coordinate {
            //lat: (y as f64 / EARTH_RADIUS_METERS).to_degrees() + base.lat,
            //lon: (x as f64 / EARTH_RADIUS_METERS).to_degrees() + base.lon,
		    lat: delta_latitude as f64 + base.lat,
            lon: delta_longitude  as f64 + base.lon
        }
    }
}

#[derive(Copy, Clone)]
struct Coordinate {
    lat: f64,
    lon: f64,
}

fn mavlink_background_process(addr: String) {
    let addr = format!("udpin:{}", addr);
    println!("Connecting to Mavlink stream: {}", addr);

    let connection = mavlink::connect(&addr).expect("Failed to connect to Mavlink stream");

    let mut gps_base = GpsBase::default();
    while STOPPED.load(Ordering::Relaxed) == false {
        match connection.recv().unwrap() {
            MavMessage::GLOBAL_POSITION_INT(data) => {
                if let Some(message) = handle_gps_data(&mut gps_base, data) {
                    println!("Sending message: {:?}", message);
                    if let Err(e) = connection.send(&message) {
                        println!("Failed to send message: {}", e);
                    }
                }
            },

            _ => {}
        }
    }
}

fn handle_gps_data(gps_base: &mut GpsBase, data: GLOBAL_POSITION_INT_DATA) -> Option<MavMessage> {
    let (dx, dy) = match gps_base.next(data.lon, data.lat) {
        Some(position) => (position[0], position[1]),
        None => return None,
    };
    
    let alt_meters = data.alt as f32 / 1e3;
    let new_position = [dx, dy, alt_meters,data.relative_alt as f32 / 1e3];
    let velocity = [data.vx as f32 / 100.0, data.vy as f32 / 100.0, data.vz as f32 / 100.0];

    let mut mavlink_data_lock = MAVLINK_DATA.lock().unwrap();
    mavlink_data_lock.position = new_position;
    mavlink_data_lock.velocity = velocity;
    mavlink_data_lock.heading = data.hdg as f32 / 100.0;
    // mavlink_data_lock.orientation = orientation;

    let target = mavlink_data_lock.next_target.take();
    drop(mavlink_data_lock);

    if let Some(target) = target {
        println!("Attempting to set new target");

        let dest_coordinate = gps_base.invert(target[0], target[1]);
        let alt = if target[2] != 0.0 { target[2] } else { alt_meters };
		let yaw = if target[3] != 7.0 { target[3] } else { -f32::NAN };
		println!("Lon: {}",dest_coordinate.lon);
		println!("Lat: {}",dest_coordinate.lat);
        return Some(generate_navigation_message(dest_coordinate.lon as f32,
            dest_coordinate.lat as f32, alt as f32, yaw as f32 ));
    }
    None
}

const MAV_CMD_DO_REPOSITION: u16 = 192;

fn generate_navigation_message(lon: f32, lat: f32, alt: f32, yaw: f32) -> MavMessage {
    MavMessage::COMMAND_LONG(COMMAND_LONG_DATA {
        param1: -1.0,
        param2: 1.0,
        param3: 0.0,
        param4: yaw, //-f32::NAN, // yaw
        param5: lat,
        param6: lon,
        param7: alt,
        command: MAV_CMD_DO_REPOSITION,
        target_system: 1,
        target_component: 0,
        confirmation: 0,
    })
}
