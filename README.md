####linux compile####
cd ~
git clone https://github.com/Cyperstereo/CyperstereoSDK.git

cd ~/CyperstereoSDK/samples
mkdir build
cd build
cmake ..
make

####run samples####

##save image and imu samples##
mkdir left
mkdir right
mkdir imu
./save_image_imu

##capture image and imu samples##
./capture_image_imu