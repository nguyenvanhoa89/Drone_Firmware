# Install FTDI Chip driver
http://www.ftdichip.com/Drivers/VCP.htm
# Follow install guide @: 
http://www.ftdichip.com/Support/Documents/AppNotes/AN_220_FTDI_Drivers_Installation_Guide_for_Linux.pdf
Open a terminal window, and enter
```
dmesg | grep FTDI
```

The output on the terminal window should contain the following:
```
[10170.987708] USB Serial support registered for FTDI USB Serial Device 
[10170.987915] ftdi_sio 9-1:1.0: FTDI USB Serial Device converter detected 
[10170.991172] usb 9-1: FTDI USB Serial Device converter now attached to ttyUSB0 
[10170.991219] ftdi_sio: v1.6.0:USB FTDI Serial Converters Driver
```

# 2. Install Arduino Studio
Open Ubuntu Software Center and search for Arduino. Alternatively, you can install via the command line by running the following in a Terminal

``` 
sudo apt-get update && sudo apt-get install arduino arduino-core  
```

# 3. Follow instrctions @: https://learn.sparkfun.com/tutorials/how-to-install-ftdi-drivers/linux


