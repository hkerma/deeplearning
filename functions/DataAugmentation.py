import PIL
import numpy as np
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import random
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib as mpl
from torchvision.datasets import CIFAR10,CIFAR100
from torch.utils.data import DataLoader,Subset

from AutoAugment import AutoAugment, Cutout



class DataAugmentation():
    def __init__(self,dataset,mode,AutoAugment,Cutout):
        self.dataset = dataset
        self.mode = mode
        self.transform_train = []
        self.transform_test = []
        self.autoaugment = AutoAugment
        self.cutout = Cutout

    def imshow(self,img):
        img = img / 2 + 0.5     #Unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    def show_transform(self,transform):
        rootdir = "./CIFAR10/"
        classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        data = CIFAR10(rootdir,train=False,download=True,transform=transform)
        loader = DataLoader(data,batch_size=10,shuffle=True,num_workers=2)

        #Get some random training images
        data= iter(loader) 
        images, labels = data.__next__()

        #Show images
        self.imshow(torchvision.utils.make_grid(images))
        print(' '.join('%5s' % classes[labels[j]] for j in range(10)))
        
    def generate_subset(self,dataset,n_classes,reducefactor,n_ex_class_init):
            nb_examples_per_class = int(np.floor(n_ex_class_init / reducefactor))
            # Generate the indices. They are the same for each class, could easily be modified to have different ones. But be careful to keep the random seed! 
            indices_split = np.random.RandomState(seed=42).choice(n_ex_class_init,nb_examples_per_class,replace=False)
            all_indices = []
            for curclas in range(n_classes):
                curtargets = np.where(np.array(dataset.targets) == curclas)
                indices_curclas = curtargets[0]
                indices_subset = indices_curclas[indices_split]
                #print(len(indices_subset))
                all_indices.append(indices_subset)
            all_indices = np.hstack(all_indices)
            
            return Subset(dataset,indices=all_indices)

    def train_transform(self,length=8):
        ###Transformation, same for CIFAR10 and CIFAR100
        self.transform_train = [transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip(),]
        if self.autoaugment:
            self.transform_train.append(AutoAugment())
        if self.cutout:
            self.transform_train.append(Cutout(length))
        self.transform_train.append(transforms.ToTensor(),)
        self.transform_train.append(transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),)
        self.transform_train = transforms.Compose(self.transform_train)
            

    def test_transform(self):
        self.transform_test = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)),])
    
    def load_data(self):
        if self.dataset == "cifar10":
            rootdir = "./CIFAR10/"
            
            ###Transformation for the training set
            self.train_transform()

            #Load CIFAR10 training dataset 
            cifar10_train = CIFAR10(rootdir,train=True,download=True,transform=self.transform_train)
            if self.mode == "test":
                cifar10_train = self.generate_subset(cifar10_train,4,5,5000)
            cifar10_train_loader = DataLoader(cifar10_train,batch_size=128, shuffle=True,num_workers=2)

            #Transformation for the test set
            self.test_transform()

            #Load CIFAR10 testing dataset
            cifar10_test = CIFAR10(rootdir,train=False,download=True,transform=self.transform_test)
            if self.mode == "test":
                cifar10_test = self.generate_subset(cifar10_test,4,5,1000)
            cifar10_test_loader = DataLoader(cifar10_test,batch_size=128,shuffle=True,num_workers=2)

            return cifar10_train_loader,cifar10_test_loader

        elif self.dataset == "cifar100":
            rootdir = "./CIFAR100/"

            ###Transformation for the training set
            self.train_transform()

            #Load CIFAR10 training dataset 
            cifar100_train = CIFAR100(rootdir,train=True,download=True,transform=self.transform_train)
            cifar100_train_loader = DataLoader(cifar100_train,batch_size=128, num_workers=2)

            #Transformation for the test set
            self.test_transform()

            #Load CIFAR10 testing dataset
            cifar100_test = CIFAR100(rootdir,train=False,download=True,transform=self.transform_test)
            cifar100_test_loader = DataLoader(cifar100_test,batch_size=128,shuffle=False,num_workers=2)
            
            return cifar100_train_loader,cifar100_test_loader

        else:
            pass


            


    
