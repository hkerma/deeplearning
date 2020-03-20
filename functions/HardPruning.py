import numpy as np
import torch.nn as nn
import math 
import torch_pruning as pruning
import torch

class HardPrunning():

    def __init__(self,model,P):
        self.model = model
        self.P = P

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
            weak1 = self.extract_min_filter(importance1,R1)
            weak2 = self.extract_min_filter(importance2,R2)
            res1.append(weak1)
            res2.append(weak2)
        conv1n,conv2n = layer[-1].conv1,layer[-1].conv2
        R1,R2 = math.floor(ratio*conv1n.out_channels),math.floor(ratio*conv2n.out_channels)
        importance1n,importance2n = self.importance_score(conv1n),self.importance_score(conv2n)
        weak_filters1n,weak_filters2n = self.extract_min_filter(importance1n,R1),self.extract_min_filter(importance2n,R2)
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
        #self.pruning_conv(layer[-1].conv2,Fn_conv2,DG)
        for i in range(N-1):
            self.pruning_conv(layer[i].conv1,Fi_conv1[i],DG)
            #self.pruning_conv(layer[i].conv2,Fi_conv2[i],DG)

    def HardPruning(self):
        DG = self.init_dependency_graph(self.model)
        r1,r2,r3 = self.P[0],self.P[1],self.P[2]
        self.pruning_wide_block(self.model.layer1,r1,DG)
        self.pruning_wide_block(self.model.layer2,r2,DG)
        self.pruning_wide_block(self.model.layer3,r3,DG)

