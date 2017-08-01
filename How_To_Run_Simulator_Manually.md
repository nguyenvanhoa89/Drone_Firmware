## 1. Run UDP fork in folder udp_fork

```
cd /home/hoa/Desktop/Github/Hoa/Drone_Firmware/Michael/udp_fork
sudo cargo run --release -- 14550 14551 14552
```

## 2. Run jMAVSim in folder Firmware

```
cd /home/hoa/Desktop/Github/Hoa/Firmware
make posix_sitl_default jmavsim 
```
All simulator configuration hardcoded in file src/me/drton/jmavsim/Simulator.java, this file should be edited before running simulator
## 3. Run QGroundControl.AppImage: 
```
Connect to port 14551 
Arm and take off to 20 meters.
```
## 4.1 Run Pulse Server If in simulated mode
```
cd /home/hoa/Desktop/Github/Michael/trackerbots_telemetry/pulse_server && 
cargo run --release -- test
```
## 4.2 Run telemetry_host in telemetry_host

```
cd /home/hoa/Desktop/Github/Michael/trackerbots_telemetry/telemetry_host && 
sudo cargo run

connect to port 14552
```
## 5. Use Matlab to control
