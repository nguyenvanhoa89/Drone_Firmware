parentdir=$(dirname `pwd`)
gnome-terminal --working-directory=$PWD/Michael/udp_fork/ -e 'cargo run --release -- 14550 14551 14552' 
gnome-terminal --working-directory=$parentdir/Firmware/ -e 'make posix_sitl_default jmavsim' 
gnome-terminal --working-directory=$PWD/ -e './QGroundControl.AppImage'
echo "Select please port 14551 on QGroundControl"
read -rsp $'Press any key to call pulse server...\n' -n1 key
gnome-terminal --working-directory=$PWD/Michael/pulse_server/ -e 'cargo run --release -- test' 
read -rsp $'Press any key to call telemetry host...\n' -n1 key
gnome-terminal --working-directory=$PWD/Michael/telemetry_host/ -e 'cargo run --release' 
read -rsp $'Press any key to call matlab program...\n' -n1 key
sh ./CallMatlab.sh


