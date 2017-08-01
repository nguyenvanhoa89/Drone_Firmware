#![feature(plugin)]
#![plugin(rocket_codegen)]

extern crate byteorder;
extern crate common;
#[macro_use] extern crate lazy_static;
extern crate mavlink;
extern crate rocket;
extern crate rocket_contrib;
extern crate serde;
#[macro_use] extern crate serde_derive;
extern crate serde_json;

mod pulse_handler;
mod mavlink_handler;

use rocket_contrib::Json;
use rocket::Rocket;

use mavlink_handler::{Telemetry, Location, MavlinkHandle};
use pulse_handler::{PulseWithTelemetry, PulseHandle};

#[get("/")]
fn get_telemetry() -> Json<Telemetry> {
    Json(mavlink_handler::get_telemetry())
}

#[post("/", data = "<location>")]
fn do_reposition(location: Json<Location>) {
    mavlink_handler::do_reposition(location.0);
}

#[get("/pulses/<index>")]
fn get_pulses(index: usize) -> Json<Vec<PulseWithTelemetry>> {
    Json(pulse_handler::get_pulses_since(index))
}

#[get("/latestpulses")]
fn get_latest_pulses() -> Json<Vec<PulseWithTelemetry>> {
    Json(pulse_handler::get_latest_pulses())
}
fn get_mavlink_addr(rocket: &Rocket) -> String {
    match rocket.config().extras.get("mavlink_addr").and_then(|x| x.as_str()) {
        Some(value) => value.into(),
        _ => "127.0.0.1:14552".into()
    }
}

fn get_pulse_server_addr(rocket: &Rocket) -> String {
    match rocket.config().extras.get("pulse_server_addr").and_then(|x| x.as_str()) {
        Some(value) => value.into(),
        _ => "127.0.0.1:11000".into()
    }
}
fn main() {
	let rocket = rocket::ignite().mount("/", routes![get_telemetry, get_pulses, do_reposition]);

    let _mavlink_handle = MavlinkHandle::new(get_mavlink_addr(&rocket));
    let _pulse_handle = PulseHandle::new(get_pulse_server_addr(&rocket));

    rocket.launch();
}
