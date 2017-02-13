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
```
## 4. Download simulator source:
```
git clone https://github.com/PX4/Firmware.git --depth=1
cd Firmware
git submodule update --init --recursive
```
## 5. Compile and run the simulator
```
make posix_sitl_default jmavsim
```
## 6. Download latest QGroundControl version:
URL: https://donlakeflyer.gitbooks.io/qgroundcontrol-user-guide/content/download_and_install.html

## 7. 
