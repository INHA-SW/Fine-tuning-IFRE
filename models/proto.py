import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F
from transformers import AdamW

from torch.optim.lr_scheduler import _LRScheduler
import math


class Proto(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, dot=False, relation_encoder=None, N=5, Q=1):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()
        self.dot = dot
        
        
        self.relation_encoder = relation_encoder
        self.hidden_size = 768
    
    
    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, rel_txt, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''

        rel_gol, rel_loc = self.sentence_encoder(rel_txt, cat=False)
        

        
        rel_loc = torch.mean(rel_loc, 1) #[B*N, D]
        #rel_rep = (rel_loc + rel_gol) /2
        #rel_rep = rel_loc
        rel_rep = torch.cat((rel_gol, rel_loc), -1)
        
        
            
        support_h, support_t,  s_loc = self.sentence_encoder(support) # (B * N * K, D), where D is the hidden size
        query_h, query_t,  q_loc = self.sentence_encoder(query) # (B * total_Q, D)
        #support = self.global_atten_entity(support_h, support_t, s_loc, rel_loc, None)
        #query = self.global_atten_entity(query_h, query_t, q_loc, None, None)
        support = torch.cat((support_h, support_t), -1)
        query = torch.cat((query_h, query_t), -1)
        
        #"""
        support = support.view(N*K, self.hidden_size*2) # (B, N, K, D)
        query = query.view(total_Q, self.hidden_size*2) # (B, total_Q, D)
        

        label = (torch.arange(0, N).unsqueeze(1).expand(N, K).reshape(-1))
        
        ##
        ###add relation into this this add a up relation dimension
        rel_rep = rel_rep.view(N, rel_gol.shape[1]*2)
        # if last_add
        # x = support.detach()
        # classifier.weight.data += rel_rep
        DIRECT_ADD = False
        if DIRECT_ADD:
            rel_rep = torch.cat([rel_rep[i].view(1, self.hidden_size*2).repeat(K, 1) for i in range(N)])
            x = torch.add(support.detach(), rel_rep.detach()).clone()
        else:
            x = torch.cat([support.detach(), rel_rep.detach()],dim=0).clone()
            rel_label = (torch.arange(0, N).unsqueeze(1).expand(N, 1).reshape(-1))#.cuda()
            label = torch.cat([label, rel_label],dim=0).clone()
        label = label.cuda()
        x = x.cuda()
        label.requires_grad = False
        
        classifier = nn.Linear(self.hidden_size * 2, N, bias=True).cuda()
        for param in classifier.parameters():
            param.requires_grad = True
        
        if N == 5 and K == 1:
            # LR = 0.05
            # ITER = 800
            # WD =5e-2
            # 95.79
            LR = 0.05
            ITER = 300
            WD =1e-1
            # 95.87
        elif N == 5 and K == 5:
            LR = 0.01
            ITER = 300
            WD =5e-2    
            #98.26
        elif N == 10 and K == 1: 
            LR = 0.03
            ITER = 500
            WD =5e-2
            #93.83
        else:
            LR = 0.05
            ITER = 300
            WD =5e-2  
            #96.80          

        optimizer = torch.optim.SGD(classifier.parameters(), lr=LR, momentum=0.9, weight_decay=WD)
        #optimizer = AdamW(classifier.parameters(), lr=LR, weight_decay = 5e-2)
        for idx in range(ITER):
            output = classifier(x)
            loss = F.cross_entropy(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        with torch.no_grad():
            logits = classifier(query)
            #pred = torch.max(logits.view(-1, N), 0)
            pred = torch.argmax(logits,dim=1)
            #pred = pred[1]
        """

        support = support.view(-1, N, K, self.hidden_size*2) # (B, N, K, D)
        query = query.view(-1, total_Q, self.hidden_size*2) # (B, total_Q, D)
        


        # Prototypical Networks 
        # Ignore NA policy
        support = torch.mean(support, 2) # Calculate prototype for each class
        ##
        ###add relation into this this add a up relation dimension
        rel_rep = rel_rep.view(-1, N, rel_gol.shape[1]*2)
        #rel_rep = self.linear(rel_rep)
        support = support + rel_rep
        
        
        
        logits = self.__batch_dist__(support, query) # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2) # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N + 1), 1)
        """
        return logits, pred

    
    
    
