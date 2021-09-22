import torch
import torch.nn as nn
from constants import *
from npn import *

class student(nn.Module):
    def __init__(self, feats1, feats2):
        super(student, self).__init__()
        self.name = "student"
        self.run1 = nn.Sequential(
            nn.Linear(feats1, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Softplus())
        self.run2 = nn.Sequential(
            nn.Linear(feats2, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Softplus())
        self.run_cat = nn.Sequential(
            nn.Linear(64, 32),
            nn.Softplus(),
            nn.Linear(32, 1),
            nn.Softplus())

    def forward(self, x1, x2):
        x1, x2 = x1.flatten(), x2.flatten()
        x1, x2 = self.run1(x1), self.run2(x2)
        x = torch.cat((x1, x2))
        x = self.run_cat(x)
        return x

class teacher(nn.Module):
    def __init__(self, feats1, feats2):
        super(teacher, self).__init__()
        self.name = "teacher"
        self.run1 = nn.Sequential(
            nn.Linear(feats1, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Softplus())
        self.run2 = nn.Sequential(
            nn.Linear(feats2, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Softplus())
        self.run_cat = nn.Sequential(
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.Softplus(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid())

    def forward(self, x1, x2):
        x1, x2 = x1.flatten(), x2.flatten()
        x1, x2 = self.run1(x1), self.run2(x2)
        x = torch.cat((x1, x2))
        x = self.run_cat(x)
        return x

class npn(nn.Module):
    def __init__(self, feats1, feats2):
        super(npn, self).__init__()
        self.name = "npn"
        self.run1 = nn.Sequential(
            NPNLinear(feats1, 32, False),
            NPNRelu(),
            NPNLinear(32, 32),
            NPNSigmoid())
        self.run2 = nn.Sequential(
            NPNLinear(feats2, 32, False),
            NPNRelu(),
            NPNLinear(32, 32),
            NPNSigmoid())
        self.run_cat = nn.Sequential(
            NPNLinear(64, 32),
            NPNRelu(),
            NPNLinear(32, 1),
            NPNSigmoid())

    def forward(self, x1, x2):
        x1, x2 = x1.view(1, -1), x2.view(1, -1)
        x1, s1 = self.run1(x1)
        x2, s2 = self.run2(x2)
        x, s = self.run_cat((torch.cat((x1, x2), dim=1), torch.cat((s1, s2), dim=1)))
        s = s.view(-1)
        return x, s
