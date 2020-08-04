
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


class CNN_preprocess(nn.Module):
    def __init__(self,width,height,channel):
        super(CNN_preprocess,self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=3,out_channels=32,kernel_size=3,stride=1,padding=(1,1))
        self.Conv2 = nn.Conv2d(in_channels=32,out_channels=128,kernel_size=5,stride=5)
        self.Conv3 = nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=3)


    def forward(self,x):
        x = self.Conv1(x)
        x = F.relu(x)
        x = self.Conv2(x)
        x = F.relu(x)
        x = self.Conv3(x)
        x = F.relu(x)
        return torch.flatten(x)

    def get_state_dim(self):
        return 64

class Actor(nn.Module):
    def __init__(self,action_dim,state_dim):
        super(Actor,self).__init__()
        self.Linear1 = nn.Linear(state_dim,128)
        self.Dropout1 = nn.Dropout(p=0.3)
        self.Linear2 = nn.Linear(128,128)
        self.Dropout2 = nn.Dropout(p=0.3)
        self.Linear3 = nn.Linear(128,action_dim)

    def forward(self,x):
        x = self.Linear1(x)
        x = self.Dropout1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        x = self.Dropout2(x)
        x = F.relu(x)
        x = self.Linear3(x)
        return F.softmax(x)

class ActorLaw(nn.Module):
    def __init__(self,action_dim,state_dim):
        super(ActorLaw,self).__init__()
        self.Linear1 = nn.Linear(state_dim,128)
        self.Dropout1 = nn.Dropout(p=0.3)
        self.Linear2 = nn.Linear(128,128)
        self.Dropout2 = nn.Dropout(p=0.3)
        self.Linear3 = nn.Linear(128,action_dim)

    def forward(self,x,rule,flag):
        x = self.Linear1(x)
        x = self.Dropout1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        x = self.Dropout2(x)
        x = F.relu(x)
        y = self.Linear3(x)
        if flag == False:
            return y
        else:
            x = torch.mul(y,rule)
            x = F.softmax(x)
            return x

class Critic(nn.Module):
    def __init__(self,state_dim):
        super(Critic,self).__init__()
        self.Linear1 = nn.Linear(state_dim, 128)
        self.Dropout1 = nn.Dropout(p=0.3)
        self.Linear2 = nn.Linear(128, 128)
        self.Dropout2 = nn.Dropout(p=0.3)
        self.Linear3 = nn.Linear(128,1)

    def forward(self,x):
        x = self.Linear1(x)
        x = self.Dropout1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        x = self.Dropout2(x)
        x = F.relu(x)
        x = self.Linear3(x)
        return x

class Centralised_Critic(nn.Module):
    def __init__(self,state_dim,n_ag):
        super(Centralised_Critic,self).__init__()
        self.Linear1 = nn.Linear(state_dim*n_ag,128)
        self.Linear2 = nn.Linear(128,1)

    def forward(self,x):
        x = self.Linear1(x)
        x = F.relu(x)
        x = self.Linear2(x)
        return x

if __name__ == "__main__":
    model_name = "pg_social"
    file_name  = "train_para/"+model_name
    agentParam = {"ifload":True,"filename": file_name,"id":"0"}
    net = torch.load(agentParam["filename"]+"pg"+agentParam["id"]+".pth",map_location = torch.device('cuda'))
    optimizer = optim.Adam(net.parameters(), lr=0.01)
