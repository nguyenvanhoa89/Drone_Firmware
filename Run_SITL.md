Change your /home/hoa/Desktop/Drone_Firmware to your Drone_Firmware location
## 1. Run UDP fork in folder io_proxy
```
cd /home/hoa/Desktop/Drone_Firmware/trackerbots_tools/io_proxy && cargo run --release -- 14550 14551 14552
```

## 2. Run jMAVSim in folder Firmware

```
cd /home/hoa/Desktop/Drone_Firmware/Firmware && make posix_sitl_default jmavsim 
```
If error, run the follow command (change compiler from gcc to clang)
```
export CC=/usr/bin/clang && export CXX=/usr/bin/clang++ && cd /home/hoa/Desktop/Drone_Firmware/Firmware && make clean && make posix_sitl_default jmavsim 
```
All simulator configuration hardcoded in file src/me/drton/jmavsim/Simulator.java, this file should be edited before running simulator
## 3. Run QGroundControl.AppImage: 
```
Connect to port 14551 
Arm and take off to 20 meters.
```
## 4.1 Run Pulse Server If in simulated mode
```
cd /home/hoa/Desktop/Drone_Firmware/trackerbots_telemetry/pulse_server && 
cargo run --release -- test
```
## 4.2 Run telemetry_host in telemetry_host
connect to port 14552
```
cd /home/hoa/Desktop/Drone_Firmware/trackerbots_telemetry/telemetry_host && 
sudo cargo run
```

## 5. Use Matlab to control
```
matlab -r "try run('/home/hoa/Desktop/Drone_Firmware/Hoa/IROS-2017/MATLAB/UAV_MultiTargets_Localisation_With_Real_Drone_Telemetry_Obs.m'); catch; end; exit"
```