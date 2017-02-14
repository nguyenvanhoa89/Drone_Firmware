#![feature(mpsc_select)]

extern crate byteorder;
extern crate common;
extern crate animal_detector;
extern crate hackrf;
#[macro_use] extern crate log;
extern crate log4rs;
extern crate mio;
extern crate serde;
 //#[macro_use] extern crate serde_derive;
extern crate serde_json;
// Read CSV file
extern crate csv;
// Pass global variable
#[macro_use] extern crate lazy_static;
use std::sync::Mutex;
//use std::sync::atomic::{ATOMIC_BOOL_INIT, AtomicBool, Ordering};
lazy_static! {
    static ref GAIN_ANGLE: Mutex<Vec<f32>> = Mutex::new(vec![]);
}
// calculate cos sin by vector
extern crate cgmath;
// Read web content
extern crate hyper;
extern crate rustc_serialize;

// Normal distribution
extern crate rand;

// Call sub programs
mod endpoint;
mod gain_control;
mod hackrf_task;
mod task;
mod test_task;
mod util;

use std::env;

fn main() {
	
	// Read from CSV File
//	let mut rdr = csv::Reader::from_file("config/Gain_Angle_Table.txt").unwrap().has_headers(false);
	let mut rdr = csv::Reader::from_file("config/3D_Directional_Gain_Pattern.txt").unwrap().has_headers(false);
	//let mut gain_angle = Vec::new();
    for record in rdr.decode() {
	/*Gain_Angle_Table
        let (theta, phi, vdb, hdb, tdb): (f32,f32,f32) = record.unwrap();
        drop(theta); drop(phi); drop(vdb); drop(hdb);
	*/
//	3D_Directional_Gain_Pattern
	let (phi, theta, tdb): (f32,f32,f32) = record.unwrap();
        drop(theta); drop(phi); 
        GAIN_ANGLE.lock().unwrap().push(tdb);
    }
//    println!("{}", GAIN_ANGLE.lock().unwrap()[26-1]);
	
	
//	std::thread::sleep(std::time::Duration::from_secs(100));
    let run_test_task = env::args().nth(1) == Some("test".into());
	
    
    log4rs::init_file("config/log_config.json", Default::default()).unwrap();
    let config = util::load_json_or_default("config/hackrf_config.json");

    if run_test_task {
    	println!("Run test");
        endpoint::start_endpoint(test_task::start_task(config));
    }
    else {
        endpoint::start_endpoint(hackrf_task::start_task(config));
    }
}


