import torch
from torch import nn
from models.layers import Conv, Hourglass, Pool

class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)

class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=128, **kwargs):
        super(PoseNet, self).__init__()
        self.pre = nn.Sequential(
            Conv(3, 64, 7, 2, bn=bn),
            Conv(64, 128, bn=bn),
            Pool(2, 2),
            Conv(128, 128, bn=bn),
            Conv(128, inp_dim, bn=bn)
        )
        self.features = nn.ModuleList( [
        nn.Sequential(
            Hourglass(4, inp_dim, bn, increase),
            Conv(inp_dim, inp_dim, 3, bn=False),
            Conv(inp_dim, inp_dim, 3, bn=False)
        ) for i in range(nstack)] )

        self.outs = nn.ModuleList( [Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)] )
        self.merge_features = nn.ModuleList( [Merge(inp_dim, inp_dim) for i in range(nstack-1)] )
        self.merge_preds = nn.ModuleList( [Merge(oup_dim, inp_dim) for i in range(nstack-1)] )

        self.nstack = nstack

    def forward(self, imgs):
        x = imgs.permute(0, 3, 1, 2)
        x = self.pre(x)
        preds = []
        for i in range(self.nstack):
            feature = self.features[i](x)
            preds.append( self.outs[i](feature) )
            if i != self.nstack - 1:
                x = x + self.merge_preds[i](preds[-1]) + self.merge_features[i](feature)
        return torch.stack(preds, 1)
