rm -rf training_data
mkdir training_data
cd training_data
wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
wget https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip
unzip vehicles.zip
unzip non-vehicles.zip
rm vehicles.zip
rm non-vehicles.zip
rm -rf __MACOSX
