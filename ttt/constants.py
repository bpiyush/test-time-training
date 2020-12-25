from os.path import join

ROOT = "/efs/test-time-training/"
DATASET_DIR = join(ROOT, "datasets")
DATASETS = {
	"CIFAR-10-C": join(DATASET_DIR, "CIFAR-10-C"),
	"CIFAR-10.1": join(DATASET_DIR, "CIFAR-10.1"),
	"CIFAR-10": join(DATASET_DIR, "CIFAR-10")
}