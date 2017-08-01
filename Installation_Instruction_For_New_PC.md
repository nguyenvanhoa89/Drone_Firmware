## 1. Install Nightly Rustc version 
URL: https://doc.rust-lang.org/book/nightly-rust.html 
```
curl -s https://static.rust-lang.org/rustup.sh | sh -s -- --channel=nightly
```
If you've got Rust installed, you can open up a shell, and type this:
```
rustc --version
```
You should see the version number, commit hash, commit date and build date:
```
ex: rustc 1.17.0-nightly (ba7cf7cc5 2017-02-11)
```
## 2. Install dependencies:
```
sudo add-apt-repository ppa:george-edison55/cmake-3.x -y
sudo apt-get update
sudo apt-get install python-argparse  python-jinja2 git-core wget zip python-empy qtcreator cmake build-essential genromfs git -y
sudo apt-get install ant protobuf-compiler libeigen3-dev libopencv-dev openjdk-8-jdk openjdk-8-jre clang-4.0 lldb-3.5 -y
sudo apt-get install libusb-1.0-0-dev -y
sudo apt-get install arduino arduino-core -y
```
## 3. Create new folder called Drone_Firmware
```
mkdir Drone_Firmware
```
## 4. Download Michael's tools 
In Drone_Firmware folder, execute below command:
```
git clone https://github.com/mchesser/trackerbots_telemetry
git clone https://github.com/mchesser/trackerbots_tools
```
## 5. Download simulator source:
In Drone_Firmware folder, execute below command:
```
git clone --recursive -j8 https://github.com/PX4/Firmware.git
cd Firmware
git submodule update --init --recursive
```
Compile and run the simulator to ensure it works well.
```
make posix_sitl_default jmavsim
```
## 6. Download latest QGroundControl version:
Download App Image @ https://s3-us-west-2.amazonaws.com/qgroundcontrol/latest/QGroundControl.AppImage

Store QGroundControl file inside the Drone_Firmware folder, then run below command to enable its executable functionality:
```
chmod +x ./QGroundControl.AppImage
sudo usermod -a -G dialout $username
sudo apt-get remove modemmanager
```
Add Port 14551 by go to Purple icon >> Comm Links >> Add:

* Name: UDP Link Port 14551
* Type: UDP
* Listenning Port: 14551
* Target Hosts: 127.0.0.1

Then press Add button to add new communication channel.

Please also uncheck UDP when startup.
## 7. Install MATLAB
* Download installer @: https://au.mathworks.com/downloads/web_downloads/download_release?release=R2016b
* Use your school email to register

## 8. Download Hoa's Github folder to run MATLAB
```
git clone https://github.cs.adelaide.edu.au/Auto-ID-Lab/Hoa.git
```