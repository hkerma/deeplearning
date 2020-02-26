# Random augmentation among the best policies for WRN 28-10 (+10% accuracy)
def rand_aug():
    return Compose([
        RandomRotate90(p=0.2),
	GaussNoise(p=0.2),
	HorizontalFlip(p=0.2),
	RandomCrop(p=0.2),
	HueSaturationValue(p=0.2),
	RandomBrightness(p=0.2),
	RandomContrast(p=0.2),
	RandomGamma(p=0.2),
	GaussianBlur(p=0.2),
	]),
	Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0)
    ])

from albumentations.pytorch.transforms import ToTensor
from albumentations.augmentations.transforms import (RandomRotate90, GaussNoise, HorizontalFlip, RandomCrop, HueSaturationValue, RandomBrightness, RandomContrast, RandomGamma, GaussianBlur, Normalize, Compose)

# Data Upload
print('\n[Phase 1] : Data Preparation')
transform_train = 
	Compose([
		Compose([
			RandomRotate90(p=0.2),
			GaussNoise(p=0.2),
			HorizontalFlip(p=0.2),
			RandomCrop(p=0.2),
			HueSaturationValue(p=0.2),
			RandomBrightness(p=0.2),
			RandomContrast(p=0.2),
			RandomGamma(p=0.2),
			GaussianBlur(p=0.2)
		], p=1.0),
		ToTensor(),
		Normalize(mean=mean[dataset], std=std[dataset], p=1.0)
	], p=1.0)
])

transform_test = Compose([
	ToTensor(),
	Normalize(mean=mean[dataset], std=std[dataset], p=1.0),
])

if(dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
elif(dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    sys.stdout.write("| ")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
