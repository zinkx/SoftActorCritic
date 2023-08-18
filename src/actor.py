import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.distributions.normal as D
import numpy as np

class PolicyNetwork(nn.Module):
    
    def __init__(self,input_dim, output_dim, hidden_dim, device, double_hidden_layer=False, use_softplus=False, init_w=3e-3):
        super(PolicyNetwork, self).__init__()
        self.device = device
        self.epsilon = 1e-10
        self.log_std_min = -20
        self.log_std_max = 2
        self.double_hidden_layer = double_hidden_layer
        self.use_softplus = use_softplus

        self.zero_scalar = torch.tensor(0).float().to(device=device)
        self.one_scalar = torch.tensor(1).float().to(device=device)
        

        self.lin1 = nn.Linear(input_dim, hidden_dim).to(device)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim).to(device)
        if self.double_hidden_layer:
            self.lin3 = nn.Linear(hidden_dim, hidden_dim).to(device)

        self.log_std_lin = nn.Linear(hidden_dim,output_dim ).to(device)
        self.log_std_lin.weight.data.uniform_(-init_w, init_w)
        self.log_std_lin.bias.data.uniform_(-init_w, init_w)
        self.mean_lin = nn.Linear(hidden_dim,output_dim ).to(device)
        self.mean_lin.weight.data.uniform_(-init_w, init_w)
        self.mean_lin.bias.data.uniform_(-init_w, init_w)

    def forward(self,state):
        x = F.relu(self.lin1(state))
        x = F.relu(self.lin2(x))
        if self.double_hidden_layer:
            x = self.lin3(x)

        mean = self.mean_lin(x)
        log_std = self.log_std_lin(x)
        if not self.use_softplus:
            log_std = torch.clamp(log_std,self.log_std_min, self.log_std_max) #prevent overflow
        return mean, log_std


    def sample_other(self, state):
        mean, log_std = self.forward(state)
        if self.use_softplus:
            std = torch.nn.functional.softplus(log_std)
        else:
            std = torch.exp(log_std)
        normal = D.Normal(self.zero_scalar, self.one_scalar)
        eps      = normal.sample((state.shape[0],1))
        action = torch.tanh(mean+ std*eps)
        log_prob = D.Normal(mean, std).log_prob(mean+ std*eps)-torch.log(1-action.pow(2)+ self.epsilon)
        log_prob = log_prob.sum(dim = 1, keepdim=True)
        
        return action, log_prob, torch.tanh(mean)
    
    def pred(self, state):
        with torch.no_grad():
            mean, log_std = self.forward(state)
            if self.use_softplus:
                std = torch.nn.functional.softplus(log_std)
            else:
                std = torch.exp(log_std)
            normal = D.Normal(0, 1)
            eps      = normal.sample()
            action = torch.tanh(mean+ std*eps)
            
            

        return action

    def exploit(self,state):
        with torch.no_grad():
            mean,log_std = self.forward(state)
            #std = torch.exp(log_std)
            #normal = D.Normal(mean, std)
            # sample actions
            #xs = normal.rsample()
            action = torch.tanh(mean)
            

        return action

