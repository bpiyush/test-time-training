from os.path import join


DATASET_DIR = "/scratch/users/piyushb/test-time-training/datasets/"
DATASETS = {
	"CIFAR-10-C": join(DATASET_DIR, "CIFAR-10-C"),
	"CIFAR-10.1": join(DATASET_DIR, "CIFAR-10.1"),
	"CIFAR-10": join(DATASET_DIR, "CIFAR-10")
}