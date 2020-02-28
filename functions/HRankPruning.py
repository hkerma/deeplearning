import torch
import torch.nn as nn
import torch_pruning as pruning


class HRank():
    def __init__(self,model,data,P):
        super(HRank, self).__init__()
        self.model = model
        self.data = data
        self.P = P

    def rank_processing(self,rank):
        processed_rank = [[l,rank[l]] for l in range(len(rank))] 
        return processed_rank

    def extract_weak_filters(self,processed_rank,P):
        importance_sorted = sorted(processed_rank,key=lambda x:x[-1])
        weak_filters = [F[0] for F in importance_sorted]
        return weak_filters[:P]
    
    def model_analysis(self):
        for _, (data,labels) in enumerate(self.data):
            output = self.model(data)

    def init_dependency_graph(self,input_w=32,input_h=32):
        DG = pruning.DependencyGraph(self.model, fake_input=torch.randn(1,3,input_w,input_h))
        return DG

    def pruning_conv(self,conv,F,DG):
        pruning_plan = DG.get_pruning_plan(conv, pruning.prune_conv, idxs=F)
        print(pruning_plan)
        pruning_plan.exec()

    def pruning_layer_1(self,layer,ratio,DG):
        N = int(((self.model.depth-4)/6))

        P1 =  math.floor(layer[-1].conv2.out_channels*ratio)
        F1 = self.extract_weak_filters(self.rank_processing(layer[-1].rank1),P1)
        self.pruning_conv(layer[-1].conv1,F1,DG)

        #P2 =  math.floor(layer[-1].conv2.out_channels*ratio)
        #F2 = self.extract_weak_filters(self.rank_processing(layer[-1].rank2),P2)
        #self.pruning_conv(layer[-1].conv2,F2,DG)

        for n in range(N-1):

            P1 =  math.floor(layer[n].conv1.out_channels*ratio)
            F1 = self.extract_weak_filters(self.rank_processing(layer[n].rank1),P1)
            self.pruning_conv(layer[n].conv1,F1,DG)

            #P2 =  math.floor(layer[n].conv2.out_channels*ratio)
            #F2 = self.extract_weak_filters(self.rank_processing(layer[n].rank2),P2)
            #self.pruning_conv(layer[n].conv2,F2,DG)
        
    def HRank(self):
        print("Sending data through the net...")
        self.model_analysis()
        print("Weak filters have been identified ! ")
        DG = self.init_dependency_graph()
        r1,r2,r3 = self.P[0],self.P[1],self.P[2]
        if r1 > 0:
            self.pruning_layer_1(self.model.layer1,r1,DG)
        else:
            pass
        if r2 > 0:
            self.pruning_layer_1(self.model.layer2,r2,DG)
        else:
            pass
        if r3 > 0:
            self.pruning_layer_1(self.model.layer3,r3,DG)
        else:
            pass
        print("The net has been pruned ! ")

    def number_of_trainable_params(self,model):
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        return sum([np.prod(p.size()) for p in model_parameters])