# 1 Edison Information
Details: https://software.intel.com/en-us/get-started-edison-linux-step2
GNURadio + HackRF: https://www.reddit.com/r/Edison/comments/49evnc/edison_image_with_gnuradio/
Loading Debian Ubilinux: https://learn.sparkfun.com/tutorials/loading-debian-ubilinux-on-the-edison
Backup + Restore Edison: http://www.instructables.com/id/BackupRestore-Intel-Edison/?ALLSTEPS


You can watch the messages that pop up on your laptop by typing
```
dmesg
```

Configure or copy: type the below command and press Enter button twice
Username: root, password: 12345678
```
sudo screen /dev/ttyUSB0 115200
```
Get edison IP address:
```
ifconfig
```
192.168.42.1

type in your laptop
```
 sudo lsusb
```
you should see a line with
```
ID 0403:6001 Future Technology Devices International, Ltd FT232 USB-Serial (UART) IC
```
type in your laptop
```
ls /dev/ttyUSB*
```
you should see
```
/dev/ttyUSB0
```
git filter-branch --index-filter 'git rm -r --cached --ignore-unmatch Edison/edison.image Edison/edison.zip Edison/iot-devkit-prof-dev-image-edison-20160606.zip' -f HEAD

# 2. Enable Access Point (AP) mode for the board
https://software.intel.com/en-us/getting-started-with-ap-mode-for-intel-edison-board
The following instructions work best on an Intel® Edison board assembled with the Arduino expansion board. 
To enter AP Mode with the Intel® Edison mini breakout board, 
you must establish a serial communication session with your board and use the command line:
```
configure_edison --enableOneTimeSetup.
```
Entering AP mode and connecting to a Wi-Fi network

    * On your board, hold down the button labeled PWR for more than 2 seconds but no longer than 7 seconds. Around 4 seconds is sufficient.

    * The LED at JS2 near the center of the board should now be blinking. It will remain blinking as long as your board is in AP mode.
    * In a few moments, a Wi-Fi network hotspot appears. Typically, its name is in the form of: Edison-xx-xx, where xx-xx is the last two places of your board's Wi-Fi mac address. This mac address is on a label within the plastic chip holder that contained the Intel® Edison chipset within the packaging. However, if you have given your board a name, the Wi-Fi hotspot has the same name.
    * When you find your board's Wi-Fi hotspot, attempt to connect.  The passcode necessary to connect is the chipset serial number, which is also on the label in the plastic chip holder beneath the mac address.  Additionally, a small white label on the Intel® Edison chipset itself also states the serial number. The passcode is case-sensitive.
    * Once you have connected to the hotspot, open a browser (preferably Firefox or Chrome) and enter Edison.local in the URL bar. The following screen displays: 

You want manuelly to switch back from AP mode in Client mode of Wifi? simply run 
configure_edison --setup again
```
systemctl stop hostapd
systemctl disable hostapd
systemctl enable wpa_supplicant
systemctl start wpa_supplicant
wpa_cli reconfigure
wpa_cli select_network wlan0
udhcpc -i wlan0
```

Install nmap to check port open or not
```
sudo apt-get install nmap
nmap -p 22 192.168.42.1
```
Then you need to connect your laptop to Edison AP
# 3. Enable Pulse Server
type the below command (copy then click middle mouse button to paste)
```
cd Drone/trackerbots_telemetry/pulse_server/
./target/release/pulse_server

```

