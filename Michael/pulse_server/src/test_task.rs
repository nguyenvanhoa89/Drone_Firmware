use std::cmp;
use std::error::Error;
use std::thread;
use std::time::Duration;
// use std::f32::consts;
use std::sync::mpsc::{channel, sync_channel, Sender, SyncSender, TrySendError};

use std::fs::File;
use std::io::{Read, BufReader};

//use std::fmt::{self, Formatter, Display};

use common::{Config, Command};
use common::signal::*;

use task::{init_task, Task, TaskHandle};

use animal_detector::Detectors;

//Global variable for gain angle
use GAIN_ANGLE;
// Read web content
use hyper::Client;
use hyper::Url;
use hyper::header::Connection;
use rustc_serialize::json;
// Normal distribution
use rand;
use rand::distributions::{Normal, IndependentSample};

#[derive(Debug, Clone,Copy, RustcDecodable, RustcEncodable)]
pub struct UavLocation {
    pub x: f32,
    pub y: f32,
    pub alt: f32,
    pub yaw: f32,
}
#[derive(Debug, Clone, RustcDecodable, RustcEncodable)]
pub struct UavWebLocation {
    pub position: Vec<f32>,
    pub heading: f32
}

#[derive(Debug, Clone, Copy)]
pub struct TargetLocation {
    pub x: f32,
    pub y: f32,
    pub alt: f32
}

pub fn start_task(config: Config) -> TaskHandle<Pulse, Command> {
    let (mut task, task_handle) = init_task();
//    println!("{}", GAIN_ANGLE.lock().unwrap()[26-1]);
//    thread::sleep(Duration::from_secs(100));
//	println!("Config is {:?}", config);
	
	
	
//	
	
	
    info!(target: "hackrf_task", "Starting test task");
    thread::spawn(move|| {
        loop {
            info!(target: "hackrf_task", "Running test");

            if let Err(e) = run_test(&mut task, config.clone()) {
                error!(target: "hackrf_task", "Test task failure: {}", e);
                thread::sleep(Duration::from_secs(10));
            }
        }
    });

    task_handle
}

fn run_test(task: &mut Task<Pulse, Command>, config: Config) -> Result<(), Box<Error>> {
    let mut test_task = TestTask {
        task: task,
        detectors: Detectors::new(&config),
    };

    let mut command = try!(test_task.task.command_receiver.recv());
    loop {
        command = match command {
            Command::Start(config) => try!(test_task.receiver_loop(config)),
            Command::Stop => try!(test_task.task.command_receiver.recv()),
            Command::Exit => break,
        };
    }
    Ok(())
}

struct TestTask<'a> {
    task: &'a mut Task<Pulse, Command>,
    detectors: Detectors,
}

impl<'a> TestTask<'a> {
    fn receiver_loop(&mut self, config: Option<Config>) -> Result<Command, Box<Error>> {
        if let Some(config) = config {
            self.detectors = Detectors::new(&config);
        }

        let command_receiver = &mut self.task.command_receiver;

        if let Ok(file) = File::open("signal.bin") {
            let (data_sender, data_receiver) = sync_channel(5);

            thread::spawn(move|| file_source(file, data_sender));

            loop {
                select! {
                    command = command_receiver.recv() => { return Ok(try!(command)); },

                    data = data_receiver.recv() => {
                        let data = try!(data);
                        for pulse in self.detectors.next(&data) {
                            try!(self.task.data_sender.send(pulse));
                        }
                    }
                }
            }
        }
        else {
        	
        	let (pulse_sender, pulse_receiver) = channel();
            thread::spawn(move|| no_source_hoa(pulse_sender));

            loop {
                select! {
                    command = command_receiver.recv() => { return Ok(try!(command)); },

                    pulse_vec = pulse_receiver.recv() => {
                    	let data = try!(pulse_vec);
                    	for pulse in data {
                    		try!(self.task.data_sender.send(pulse));
                    	} 
                    }
                    
                }
            }
            
//        	
//            let (pulse_sender, pulse_receiver) = channel();
//            thread::spawn(move|| no_source(pulse_sender));
//
//            loop {
//                select! {
//                    command = command_receiver.recv() => { return Ok(try!(command)); },
//
//                    pulse = pulse_receiver.recv() => {
//                        try!(self.task.data_sender.send(try!(pulse)));
//                    }
//                }
//            }
            
        }
    }
}

const FRAME_SIZE: usize = 4_000_000;

fn file_source(file: File, sender: SyncSender<Vec<u8>>) {
	println!("File source called");
    let mut reader = BufReader::new(file);
    let data = {
        let mut buffer = vec![];
        reader.read_to_end(&mut buffer).unwrap();
        buffer
    };

    let mut index = 0;

    let mut frame = vec![0; FRAME_SIZE];

    loop {
        let read_size = cmp::min(data.len() - index, FRAME_SIZE);
        frame[0..read_size].copy_from_slice(&data[index..(index + read_size)]);

        if read_size < FRAME_SIZE {
            index = FRAME_SIZE - read_size;
            frame[read_size..FRAME_SIZE].copy_from_slice(&data[0..index]);
        }
        else {
            index += read_size;
        }

        match sender.try_send(frame.clone()) {
            Ok(()) => {},
            Err(TrySendError::Full(_)) => warn!(target: "hackrf_task", "Sample dropped"),
            Err(_) => break,
        }

        thread::sleep(Duration::from_secs(1));
    }
}
//
//fn no_source(pulse_sender: Sender<Pulse>) {
//    const PULSE_A_PERIOD: f32 = 1.0;
//    const PULSE_B_PERIOD: f32 = 0.98;
//
//    let mut pulse_a_time = 0.0;
//    let mut pulse_b_time = 0.0;
//
//    loop {
//        let now = Timestamp::now();
//
//        if pulse_a_time > PULSE_A_PERIOD {
//            if pulse_sender.send(Pulse { freq: 148.0, signal_strength: 1.0, gain: 0, timestamp: now }).is_err() {
//                break;
//            }
//            pulse_a_time -= PULSE_A_PERIOD;
//        }
//
//        if pulse_b_time > PULSE_B_PERIOD {
//            if pulse_sender.send(Pulse { freq: 150.0, signal_strength: 1.0, gain: 0, timestamp: now }).is_err() {
//                break;
//            }
//            pulse_b_time -= PULSE_B_PERIOD;
//        }
//
//        thread::sleep(Duration::from_millis(100));
//
//        pulse_a_time += 0.1;
//        pulse_b_time += 0.1;
//	}
//}


fn no_source_hoa(pulse_sender: Sender<Vec<Pulse>>) {
    loop {
//    	println!("No source called");
		thread::sleep(Duration::from_millis(100));
        let now = Timestamp::now();
        //Intialize targets
		let target1  = TargetLocation {x: 50.0, y: 100.0, alt: 0.0};
		let target2  = TargetLocation {x: 200.0, y: 300.0, alt: 0.0};
		let target3  = TargetLocation {x: 400.0, y: 50.0, alt: 0.0};
		// Get UAV location
        let uav: UavLocation = get_uav_location();
        // Gaussian distribution
        let normal = Normal::new(0.0, 6.0); // mean:0; sigma: 6;	
		/*
		// Initilaize observation
		let pt_w:f32 = 0.5e-3;
		let pt = 10.0*(pt_w).log(10.0);//dBm
		let f:f32 = 146e6;
		let c:f32 = 299792458.0;
		let lambda:f32 = c/f;
		let gt:f32 = 0.0;   // dBm
		let gr:f32 = -10.0; //dBm %Previous: -15
		let loss:f32 = 10.0; //dBm 
		// Calculate Observation		
		let obs1 = friis_2model(pt, gt, gr, lambda, loss, target1,uav,get_antenna_gain(target1, uav)) + normal.ind_sample(&mut rand::thread_rng()) as f32 ;
		let obs2 = friis_2model(pt, gt, gr, lambda, loss, target2,uav,get_antenna_gain(target2, uav)) + normal.ind_sample(&mut rand::thread_rng()) as f32;
		let obs3 = friis_2model(pt, gt, gr, lambda, loss, target3,uav,get_antenna_gain(target3, uav)) + normal.ind_sample(&mut rand::thread_rng())  as f32;
		
		*/
		// Observation with ref information
		let a_ref:f32 = -10.65; //12.67 if Gain_Angle_Table, -10.65 if 3D_Directional_Gain_Pattern
		let d_ref:f32 = 40.0;
		// Calculate Observation		
		let obs1 = friis_with_ref(a_ref, d_ref, target1,uav,get_antenna_gain(target1, uav)) + normal.ind_sample(&mut rand::thread_rng()) as f32 ;
		let obs2 = friis_with_ref(a_ref, d_ref, target2,uav,get_antenna_gain(target2, uav))  + normal.ind_sample(&mut rand::thread_rng()) as f32;
		let obs3 = friis_with_ref(a_ref, d_ref, target3,uav,get_antenna_gain(target3, uav))  + normal.ind_sample(&mut rand::thread_rng())  as f32;	
		// Update pulse		
        let pulse1 = Pulse { freq: 148.0, signal_strength: obs1, gain: 48, timestamp: now };
        let pulse2 = Pulse { freq: 152.0, signal_strength: obs2, gain: 40, timestamp: now };
        let pulse3 = Pulse { freq: 150.0, signal_strength: obs3, gain: 32, timestamp: now };
        let pulse_vec = vec![pulse1, pulse2, pulse3];
        println!("Pulse: {:?}", pulse_vec);
        if pulse_sender.send(pulse_vec).is_err() {
            break;
        }
        thread::sleep(Duration::from_millis(900));
    }
}

fn get_uav_location() -> UavLocation {
	let client = Client::new();
	let url = match Url::parse("http://localhost:8000") {
        Ok(url) => url,
        Err(_) => panic!("Uh oh."),
    };
	let mut res = client.get(url).header(Connection::close()).send().unwrap();
//	assert_eq!(res.status, hyper::Ok);
	let mut body = String::new();
    res.read_to_string(&mut body).unwrap();
    let uav_decoded: UavWebLocation = json::decode(&body).unwrap();
    let  uav_location = UavLocation {
	    x : uav_decoded.position[0],
	    y : uav_decoded.position[1],
	    alt : uav_decoded.position[3],
	    yaw : uav_decoded.heading.to_radians()
    };
    return uav_location;
//    println!("{:?}", decoded);
}

fn get_antenna_gain(target: TargetLocation, uav: UavLocation) -> f32 {
	use cgmath::prelude::*;
	use cgmath::Vector2;
	let v1 = Vector2::new(target.x, target.y) - Vector2::new(uav.x, uav.y);
	let v2 = Vector2::new(uav.yaw.sin(), uav.yaw.cos()) ;
	let resolution:f32 = 15.0; // 15 if Gain_Angle_Table, 15 if 3D_Directional_Gain_Pattern
	let phi_in_rad : f32 = v1.angle(v2.into()).0;
	let mut phi_in_degrees = phi_in_rad.to_degrees() as f32;
	if phi_in_degrees < 0.0 {
		phi_in_degrees = phi_in_degrees + 360.0;
	}
	let phi_index : f32 = (phi_in_degrees/resolution).floor()  * resolution ;
	let theta_in_degrees : f32 = (((target.alt - uav.alt).abs()/calculate_uav_target_distance(target, uav).sqrt()).acos()).to_degrees();
	let theta_index: f32 = (theta_in_degrees/resolution).floor()  * resolution ;
	let gain_index: usize = ((360.0/resolution + 1.0) * (theta_index/resolution) + phi_index/resolution ) as usize;
//	println!("phi: {}, theta: {}", phi_in_degrees, theta_in_degrees);
	GAIN_ANGLE.lock().unwrap()[gain_index]
}

fn calculate_uav_target_distance(target: TargetLocation, uav: UavLocation) -> f32{
	(target.x - uav.x) * (target.x - uav.x) + (target.y - uav.y) * (target.y - uav.y) + (target.alt - uav.alt) * (target.alt - uav.alt)
}
/*
fn friis_2model(pt: f32, gt: f32, gr: f32, lambda: f32, loss: f32, target: TargetLocation ,uav: UavLocation, antenna_gain: f32) -> f32 {
	let ht:f32 = target.alt+1.0; 
	let hr:f32 = uav.alt;
	let d:f32 = (target.x - uav.x)*(target.x - uav.x) + (target.y - uav.y)*(target.y - uav.y);
	let l:f32 = (d+ (ht+hr)*(ht+hr)).sqrt() -  (d + (ht-hr)*(ht-hr)).sqrt();
	let pi:f32 = consts::PI;
	let phi:f32 =  (2.0*l*pi/lambda) as f32;
//	println!("pt: {}, gt: {}, gr: {}, lamda: {}, loss : {}, d: {}, phi: {}, gain: {}, l :{}, ht: {}, hr: {}", pt, gt, gr, lambda, loss, d, phi, antenna_gain,l, ht, hr);
	let pr:f32 = pt + gt + gr - loss + antenna_gain- 10.0*(4.0*pi*pi).log(10.0) -10.0*(d).log(10.0) + 20.0*(((phi/2.0).sin()).abs()).log(10.0) + 20.0*(lambda).log(10.0);
	return pr; 
}
*/
fn friis_with_ref (a_ref: f32,d_ref: f32, target: TargetLocation ,uav: UavLocation, antenna_gain: f32) -> f32 {
	let d = calculate_uav_target_distance(target, uav).sqrt();
	a_ref - 10.0*2.0*(d).log(10.0) + 10.0*2.0*(d_ref).log(10.0) + antenna_gain
}

