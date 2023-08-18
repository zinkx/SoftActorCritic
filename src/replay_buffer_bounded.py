import numpy as np#
import torch
class Buffer():
    def __init__(self, dim_r,dim_s,dim_a, max_size, device):
        self.dim_a = dim_a
        self.dim_r = dim_r
        self.dim_s = dim_s
        self.pin_memory = not(device.type == 'cpu')
        self.Rewards = torch.zeros([max_size,dim_r])
        self.States = torch.zeros([max_size,dim_s])
        self.Next_States = torch.zeros([max_size,dim_s])
        self.Done = torch.zeros([max_size,1])
        self.Actions = torch.zeros([max_size,dim_a])
        if self.pin_memory:
            self.Rewards = self.Rewards.pin_memory()
            self.States = self.States.pin_memory()
            self.Next_States = self.Next_States.pin_memory()
            self.Done = self.Done.pin_memory()
            self.Actions = self.Actions.pin_memory()
        self.size = 0
        self.max_size = max_size
        self.current_idx = 0
        self.device = device
        
    def add_Batch(self,s,a,r,s_,d):
        R = torch.tensor(r).float()
        A = torch.from_numpy(a).float()
        S = torch.from_numpy(s).float()
        S_ = torch.from_numpy(s_).float()
        D = torch.tensor(d).float()
        idx = (self.current_idx) % self.max_size
        self.Rewards[idx, :] = R
        self.Actions[idx, :] = A
        self.States[idx, :] = S
        self.Done[idx, :] = D
        self.Next_States[idx, :] = S_
        self.current_idx = (self.current_idx + 1) % self.max_size
        self.size = min(self.max_size, self.size + 1)
        
    def get_Sample(self):
        i = np.random.randint(low = 0, high=self.size+1,size = 1)
        return self.States[i], self.Actions[i], self.Rewards[i],self.Next_States[i],self.Done[i]

    def resize(self, new_size):
        old_size = self.max_size
        Rewards = torch.zeros([new_size,self.dim_r])
        Rewards[:old_size, :] = self.Rewards
        States = torch.zeros([new_size,self.dim_s])
        States[:old_size, :] = self.States
        Next_States = torch.zeros([new_size,self.dim_s])
        Next_States[:old_size, :] = self.Next_States
        Done = torch.zeros([new_size,1])
        Done[:old_size, :] = self.Done
        Actions = torch.zeros([new_size,self.dim_a])
        Actions[:old_size, :] = self.Actions

        self.Rewards = Rewards
        self.States = States
        self.Next_States = Next_States
        self.Done = Done
        self.Actions = Actions

        if self.pin_memory:
            self.Rewards = self.Rewards.pin_memory()
            self.States = self.States.pin_memory()
            self.Next_States = self.Next_States.pin_memory()
            self.Done = self.Done.pin_memory()
            self.Actions = self.Actions.pin_memory()

        self.max_size = new_size
    
    def get_Sample_batch(self,batch):
        if batch > self.size:
            batch = self.size
        idx = np.random.choice(range(self.size), size = batch, replace = False)
        R = self.Rewards[idx].to(self.device)
        A = self.Actions[idx].to(self.device)
        S = self.States[idx,:].to(self.device)
        S_ = self.Next_States[idx,:].to(self.device)
        Done = self.Done[idx].to(self.device)
        
        return S,A,R,S_,Done

    def get_ordered_Batch(self,batch):
        if batch > self.size:
            batch = self.size
        idx = np.random.choice(range(self.size), size = 1, replace = False)
        idx+=1
        R = self.Rewards[idx].to(self.device)
        A = self.Actions[idx].to(self.device)
        S = self.States[idx,:].to(self.device)
        S_ = self.Next_States[idx,:].to(self.device)
        Done = self.Done[idx].to(self.device)
        
    def clear_Buffer(self):
        self.Rewards = np.empty(self.dim_r)
        self.States = np.empty(self.dim_s)
        self.Next_States = np.empty(self.dim_s)
        self.Done = np.empty(1)
        self.Actions = np.empty(self.dim_a)
        self.size = 0