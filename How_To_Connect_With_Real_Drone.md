## 1. Run IO Proxy Tool developed by Michael:

```
cd /home/hoa/Desktop/Github/Michael/trackerbots_tools/io_proxy
cargo run --release -- serial:/dev/ttyUSB0 14551 14552
```
## 2. Run QGroundControl.AppImage:
```
cd /home/hoa/Desktop/Github/Hoa/Drone_Firmware
./QGroundControl.AppImage
```
connect to port 14551
## 3 Run Pulse Server in Edison board
* Turn on edison board
* Entering AP mode and connecting to a Wi-Fi network
	* Connect through USB serial
	```
	sudo screen /dev/ttyUSB0 115200
	configure_edison --enableOneTimeSetup.
	```

	* On your board, hold down the button labeled PWR for more than 2 seconds but no longer than 7 seconds. Around 4 seconds is sufficient.

	* The LED at JS2 near the center of the board should now be blinking. It will remain blinking as long as your board is in AP mode.
	* In a few moments, a Wi-Fi network hotspot appears. Typically, its name is in the form of: Edison-xx-xx, where xx-xx is the last two places of your board's Wi-Fi mac address. This mac address is on a label within the plastic chip holder that contained the Intel® Edison chipset within the packaging. However, if you have given your board a name, the Wi-Fi hotspot has the same name.
	* When you find your board's Wi-Fi hotspot, attempt to connect.  The passcode necessary to connect is the chipset serial number, which is also on the label in the plastic chip holder beneath the mac address.  Additionally, a small white label on the Intel® Edison chipset itself also states the serial number. The passcode is case-sensitive.
	* Once you have connected to the hotspot, open a browser (preferably Firefox or Chrome) and enter Edison.local in the URL bar. The following screen displays:
* Connect your laptop to Edison wifi (eg. Wifi name: football, pwd: 12345678)
* Open Putty and Run pulse server as below:
	```
	systemctl start edison_pulse_server.service             # Enable the service without restarting
	systemctl restart edison_pulse_server.service			# Restart the service
	journalctl --no-pager -u edison_pulse_server.service    # Show log to ensure the service is working
	```
* Note: Don't use Android/iPhone connect to your laptop as Internet Ethernet connection.
## 4 Run telemetry_host in telemetry_host

```
cd /home/hoa/Desktop/Github/Hoa/Drone_Firmware/Michael/telemetry_host
cargo run --release
```
connect to port 14552

## 5. Use Matlab to control
* 1/4 of search area:
	```
	matlab -nodisplay -nodesktop -r "try run('/home/hoa/Desktop/Github/Hoa/Drone_Firmware/Matlab/UAV_MultiTargets_Localisation_With_Real_Drone_Telemetry_Obs.m'); catch; end; exit"

	```
* Rr drone start at center
	```
	matlab -nodisplay -nodesktop -r "try run('/home/hoa/Desktop/Github/Hoa/Drone_Firmware/Matlab/UAV_MultiTargets_Localisation_With_Real_Drone_Telemetry_Obs_At_Center.m'); catch; end; exit"
	```
