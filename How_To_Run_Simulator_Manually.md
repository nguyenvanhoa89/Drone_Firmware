## 1. Run UDP fork in folder udp_fork

```
sudo cargo run --release -- 14550 14551 14552
```

## 2. Run jMAVSim in folder Firmware

```
make posix_sitl_default jmavsim
```
All simulator configuration hardcoded in file src/me/drton/jmavsim/Simulator.java, this file should be edited before running simulator
## 3. Run QGroundControl.AppImage: 
connect to port 14551
## 4.1 Run Pulse Server If in simulated mode
cargo run --release -- test
## 4.2 Run telemetry_host in telemetry_host

```
sudo cargo run
```
connect to port 14552

## 5. Use Matlab to control
