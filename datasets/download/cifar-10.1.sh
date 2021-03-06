DATASET_DIR=/efs/test-time-training/datasets
folder=$DATASET_DIR/CIFAR-10.1/raw

mkdir -p $folder

# download the data
path=$folder/cifar10.1_v4_data.npy
wget -O $path https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_data.npy
path=$folder/cifar10.1_v6_data.npy
wget -O $path https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_data.npy

# download the labels
path=$folder/cifar10.1_v4_labels.npy
wget -O $path https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v4_labels.npy
path=$folder/cifar10.1_v6_labels.npy
wget -O $path https://github.com/modestyachts/CIFAR-10.1/raw/master/datasets/cifar10.1_v6_labels.npy