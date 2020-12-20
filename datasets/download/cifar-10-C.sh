folder=/scratch/users/piyushb/test-time-training/datasets/CIFAR-10-C

# download .tar file
path=$folder/CIFAR-10-C.tar
wget -O $path https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1

# untar and store in raw/
tar -xvf $folder/CIFAR-10-C.tar -C $folder/raw/
mv $folder/raw/CIFAR-10-C/* $folder/raw/.
rm -rf $folder/raw/CIFAR-10-C/