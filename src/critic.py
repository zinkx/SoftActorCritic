import torch.nn as nn
import torch.nn.functional as F
import torch

#Q network

class SoftQNetwork(nn.Module):
    def __init__(self,input_dim, output_dim, hidden_dim, device, double_hidden_layer=False, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        self.double_hidden_layer = double_hidden_layer
        
        self.lin1 = nn.Linear(input_dim+output_dim, hidden_dim).to(device)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        if self.double_hidden_layer:
            self.h2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        self.lin3 = nn.Linear(hidden_dim, output_dim).to(device)

        self.lin3.weight.data.uniform_(-init_w, init_w)
        self.lin3.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state,action):

        x=torch.cat([state,action],1) 
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        if self.double_hidden_layer:
            x = F.relu(self.h2(x))
        x = self.lin3(x)
        return x

    