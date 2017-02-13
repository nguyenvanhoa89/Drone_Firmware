gnome-terminal --working-directory=$PWD/Michael/udp_fork/ -e 'cargo run --release -- 14550 14551 14552' 
gnome-terminal --working-directory=$PWD/Firmware/ -e 'make posix_sitl_default jmavsim' 
gnome-terminal --working-directory=$PWD/ -e './QGroundControl.AppImage'
echo "Select please port 14551 on QGroundControl"
sleep 60
gnome-terminal --working-directory=$PWD/Michael/pulse_server/ -e 'cargo run --release -- test' 
sleep 10
gnome-terminal --working-directory=$PWD/Michael/telemetry_host/ -e 'cargo run --release' 
