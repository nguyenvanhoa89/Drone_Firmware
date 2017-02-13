## 1. Clone current directory
```
git clone https://github.com/nguyenvanhoa89/Drone_Firmware.git
```

## 2. Install Nightly Rustc version 
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
rustc 1.17.0-nightly (ba7cf7cc5 2017-02-11)
```
## 3. Install dependencies:
```
sudo add-apt-repository ppa:george-edison55/cmake-3.x -y
sudo apt-get update
sudo apt-get install python-argparse git-core wget zip python-empy qtcreator cmake build-essential genromfs -y
sudo apt-get install ant protobuf-compiler libeigen3-dev libopencv-dev openjdk-8-jdk openjdk-8-jre clang-3.5 lldb-3.5 -y
sudo apt-get install libusb-1.0-0-dev -y
```
## 4. Download simulator source:
```
git clone https://github.com/PX4/Firmware.git --depth=1
cd Firmware
git submodule update --init --recursive
```
## 5. Compile and run the simulator
Ensure it works well.
```
make posix_sitl_default jmavsim
```
## 6. Download latest QGroundControl version:
URL: https://donlakeflyer.gitbooks.io/qgroundcontrol-user-guide/content/download_and_install.html

App Image URL: https://s3-us-west-2.amazonaws.com/qgroundcontrol/latest/QGroundControl.AppImage

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

Go to main map, arm and take off UAV to pre-defined altitudes.
## 7. Download and Install MAVProxy
URL: http://ardupilot.github.io/MAVProxy/html/getting_started/download_and_installation.html
First, a few pre-requisite packages need to be installed:
```
sudo apt-get install python-dev python-opencv python-wxgtk3.0 python-pip python-matplotlib python-pygame python-lxml -y
```
Then download and install MAVProxy via Pypi. Prerequisites will be automatically downloaded too. Note a sudo may be required in some circumstances if the install generates errors:
```
pip install MAVProxy
```
If not already set, MAVProxy needs to be on the system path:
```
echo "export PATH=$PATH:$HOME/.local/bin" >> ~/.bashrc
```
The user permissions may also need to be changed to allow access to serial devices:
```
sudo adduser <username> dialout
```
Use MaVProxy to configure UDP Port:
```
mavproxy.py --master=udp:127.0.0.1:14550 --out=udp:127.0.0.1:14551 --out=udp:127.0.0.1:14552
```
## 8. Run Simulator.sh
Can run auto by typing the below command inside Drone_Firmware folder:
```
chmod +x ./RunSimulator.sh
chmod +x ./CallMatlab.sh
chmod +x ./CallMatlab.sh
./Closs_all_terminals.sh
```
Or manually through instructions @: https://github.com/nguyenvanhoa89/Drone_Firmware/blob/master/How_To_Run_Simulator_Manually.md
