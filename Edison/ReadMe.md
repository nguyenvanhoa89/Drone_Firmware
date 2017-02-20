# Edison Information
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
129.127.146.175

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

