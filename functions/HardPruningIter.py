import numpy as np
import torch.nn as nn
import math 
import torch_pruning as pruning
import torch
import torch.optim as optim
from torch.autograd import Variable
import sys 
import os 


import torch
from ptflops import get_model_complexity_info


class HardPrunningIter():

    def __init__(self,model,P):
        self.model = model
        self.P = P
        self.best_acc = [0]
        self.net_weights = [0]
        self.count = [0]*3
        self.lenght = [self.model.widen_factor*16, self.model.widen_factor*32, self.model.widen_factor*64]
        self.max_iter = int(max(np.multiply(self.lenght,self.P))/5)

    def importance_score(self,conv):
        F = conv.out_channels
        importance = []
        for i in range(F):
            filt = conv.weight[i]
            score = filt.norm()
            importance.append([i,score])
        return importance

    def extract_min_filter(self,importance,ratio):
        importance_sorted = sorted(importance,key=lambda x:x[-1])
        res = [x[0] for x in importance_sorted]
        return res[:ratio]

    def extract_min_filter_from_wide_block(self,layer,ratio):
        res1,res2,resn = [],[],[]
        N = int(((self.model.depth-4)/6)-1)

        for n in range(N):
            conv1 = layer[n].conv1
            conv2 = layer[n].conv2
            R1 = math.floor(ratio*conv1.out_channels)
            R2 = math.floor(ratio*conv2.out_channels)
            importance1 = self.importance_score(conv1)
            importance2 = self.importance_score(conv2)
            weak1 = self.extract_min_filter(importance1,5)
            weak2 = self.extract_min_filter(importance2,5)
            res1.append(weak1)
            res2.append(weak2)
        conv1n,conv2n = layer[-1].conv1,layer[-1].conv2
        R1,R2 = math.floor(ratio*conv1n.out_channels),math.floor(ratio*conv2n.out_channels)
        importance1n,importance2n = self.importance_score(conv1n),self.importance_score(conv2n)
        weak_filters1n,weak_filters2n = self.extract_min_filter(importance1n,5),self.extract_min_filter(importance2n,5)
        resn.append(weak_filters1n)
        resn.append(weak_filters2n)
        return res1,res2,resn
      
    def init_dependency_graph(self,model,input_w=32,input_h=32):
        DG = pruning.DependencyGraph(self.model, fake_input=torch.randn(1,3,input_w,input_h).cuda())
        return DG

    def pruning_conv(self,conv,F,DG):
        pruning_plan = DG.get_pruning_plan(conv, pruning.prune_conv, idxs=F)
        pruning_plan.exec()

    def number_of_trainable_params(self,model):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])
    
    def pruning_wide_block(self,layer,ratio,DG):
        weak_filters = self.extract_min_filter_from_wide_block(layer,ratio)
        N = int(((self.model.depth-4)/6))
        Fi_conv1,Fi_conv2,Fn_conv1,Fn_conv2  = weak_filters[0],weak_filters[1],weak_filters[2][0],weak_filters[2][1]
        self.pruning_conv(layer[-1].conv1,Fn_conv1,DG)
        self.pruning_conv(layer[-1].conv2,Fn_conv2,DG)
        for i in range(N-1):
            self.pruning_conv(layer[i].conv1,Fi_conv1[i],DG)
            #self.pruning_conv(layer[i].conv2,Fi_conv2[i],DG)

    def HardPruning(self):
        DG = self.init_dependency_graph(self.model)
        r1,r2,r3 = self.P[0],self.P[1],self.P[2]
        if r1 > 0 and (self.count[0] < math.floor(self.lenght[0]*self.P[0])): 
            self.count[0] += 1
            self.pruning_wide_block(self.model.layer1,r1,DG)
        if r2 > 0 and (self.count[1] < math.floor(self.lenght[1]*self.P[1])):
            self.count[1] += 1
            self.pruning_wide_block(self.model.layer2,r2,DG)
        if r3 > 0 and (self.count[2] < math.floor(self.lenght[2]*self.P[2])):
            self.count[2] += 1
            self.pruning_wide_block(self.model.layer3,r3,DG)
    
    def learning_rate(self,epoch,init):
        optim_factor = 0
        if (epoch > 3):
            optim_factor = 3
        elif(epoch > 2):
            optim_factor = 2
        elif(epoch > 1):
            optim_factor = 1
        return init*math.pow(0.2,optim_factor)


    def pruning_and_training(self, testloader, trainloader, batch_size = 128, epoch = 1, lr = 0.001):
        for it in range(self.max_iter):
            best_acc = -1000
            print('\n[1] PRUNING | ITER : {}/{}-----------------------------------------------------------'.format(it+1,self.max_iter))
            print('\n=> Pruning Net... | Layer1 : {}% Layer2 : {}% Layer3 : {}%'.format(self.P[0]*100,self.P[1]*100,self.P[2]*100))
            self.HardPruning()
            self.model.train()
            flops,params = get_model_complexity_info(self.model,(3, 32, 32), as_strings=True,print_per_layer_stat=False)
            print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))
            print('\n[2] FINE TUNING----------------------------------------------------------------------')
            for e in range(epoch):
                train_loss = 0
                correct = 0
                total = 0
                optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate(e,lr), momentum=0.9)
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
                    sys.stdout.write('Trainable params [{}]'.format(params))
                    sys.stdout.write('| Iteration [%3d] Epoch [%3d/%3d] Iter [%3d/%3d] LR [%3d] \t\tLoss: %.4f Acc@1: %.3f%%'%(it+1,e + 1, epoch, batch_idx+1,391, self.learning_rate(e,lr),loss.item(), 100.*correct/total))
                    sys.stdout.flush()

                self.model.eval()
                self.model.training = False
                test_loss = 0
                correct = 0
                total = 0
                criterion = nn.CrossEntropyLoss()
                with torch.no_grad():
                    for batch_idx, (inputs, targets) in enumerate(testloader): 
                    	  inputs, targets = inputs.cuda(), targets.cuda()
                    	  inputs, targets = Variable(inputs), Variable(targets)
                    	  outputs = self.model(inputs)
                    	  loss = criterion(outputs, targets)
                    	  test_loss += loss.item()
                    	  predicted = torch.max(outputs.data, 1)[1]
                    	  total += targets.size(0)
                    	  correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
                    acc = 100.*correct/total
                    print('\n | Test {} '.format(acc))
                    if acc > best_acc:
                    	  print('| New Best Accuracy...\t\t\tTop1 = %.2f%%' %(acc)) 
                    	  print('| Saving Pruned Model...')
                    	  torch.save(self.model,"wide_resnet_iter_hard.pth")
                    	  best_acc = acc
            self.best_acc.append(best_acc.item())
            self.net_weights.append(self.number_of_trainable_params(self.model))