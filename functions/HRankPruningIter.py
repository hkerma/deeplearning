import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch_pruning as pruning
#from WideResnet_HRank import Wide_ResNet
from DataAugmentation import DataAugmentation
from AutoAugment import AutoAugment, Cutout

import os
import sys

#print(model_test)
class HRank():
    def __init__(self,model,data,P,r):
        super(HRank, self).__init__()
        self.model = model
        self.data = data
        self.r = r
        self.P = P
        #Suppose that our net has three layers
        self.count = [0]*3
        self.lenght = [self.model.widen_factor*16, self.model.widen_factor*32, self.model.widen_factor*64]
        self.max_iter = int((1/self.r)*int(max(np.multiply(self.lenght,self.P))))

    def rank_processing(self,rank):
        processed_rank = [[l,rank[l]] for l in range(len(rank))] 
        return processed_rank

    def extract_weak_filters(self,processed_rank,P):
        importance_sorted = sorted(processed_rank,key=lambda x:x[-1])
        weak_filters = [F[0] for F in importance_sorted]
        return weak_filters[:P]
    
    def model_analysis(self):
        for _, (data,labels) in enumerate(self.data):
            data = data.cuda()
            labels = labels.cuda()
            output = self.model(data)

    def init_dependency_graph(self,input_w=32,input_h=32):
        DG = pruning.DependencyGraph(self.model, fake_input=torch.randn(1,3,input_w,input_h).cuda())
        return DG

    def pruning_conv(self,conv,F,DG):
        pruning_plan = DG.get_pruning_plan(conv, pruning.prune_conv, idxs=F)
        pruning_plan.exec()

    def pruning_layer_1(self,layer,DG):
        N = int(((self.model.depth-4)/6))
        F1 = self.extract_weak_filters(self.rank_processing(layer[-1].rank1),self.r)
        self.pruning_conv(layer[-1].conv1,F1,DG)

        for n in range(N-1):
            F1 = self.extract_weak_filters(self.rank_processing(layer[n].rank1),self.r)
            self.pruning_conv(layer[n].conv1,F1,DG)
        
    def HRank(self):
        print("Sending data through the net...")
        self.model_analysis()
        print("Weak filters have been identified ! ")
        DG = self.init_dependency_graph()
        r1,r2,r3 = self.P[0],self.P[1],self.P[2]
        if r1 > 0 and (self.count[0] < math.floor(self.lenght[0]*self.P[0])):
            self.count[0] += self.r
            self.pruning_layer_1(self.model.layer1,DG)
        else:
            pass
        if r2 > 0 and (self.count[1] < math.floor(self.lenght[1]*self.P[1])):
            self.count[1] += self.r
            self.pruning_layer_1(self.model.layer2,DG)
        else:
            pass
        if r3 > 0 and (self.count[2] < math.floor(self.lenght[2]*self.P[2])):
            self.count[2] += self.r
            self.pruning_layer_1(self.model.layer3,DG)
        else:
            pass
        print("The net has been pruned ! ")

    def number_of_trainable_params(self,model):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])

    def pruning_and_training(self, trainloader, batch_size = 128, epoch = 2, lr = 0.01):
        for it in range(self.max_iter):
            self.HRank()
            self.model.train()
            for e in range(epoch):
                train_loss = 0
                correct = 0
                total = 0
                optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9)
                criterion = nn.CrossEntropyLoss()
                for batch_idx, (inputs,targets) in enumerate(trainloader):
                    inputs, targets = inputs.cuda(),targets.cuda()
                    optimizer.zero_grad()
                    inputs, targets = Variable(inputs), Variable(targets)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    total += targets.size(0)
                    predicted = torch.max(outputs.data, 1)[1]
                    train_loss += loss.item()
                    correct += predicted.eq(targets.data).cpu().sum()
                    sys.stdout.write('\r')
                    sys.stdout.write('Trainable params [{}]'.format(self.number_of_trainable_params(self.model)))
                    sys.stdout.write('|Iteration [%3d] Epoch [%3d/%3d] Iter[%3d]\t\tLoss: %.4f Acc@1: %.3f%%'%(it,e, epoch, batch_idx+1, loss.item(), 100.*correct/total))
                    sys.stdout.flush()
