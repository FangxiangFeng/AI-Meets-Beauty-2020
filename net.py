import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from pooling import *
from collections import OrderedDict

__all__=['L2N', 'resnet50bt', 'resnet101bt']

#---------------feature extraction------------------#

class L2N(nn.Module):

    def __init__(self, eps=1e-6):
        super(L2N,self).__init__()
        self.eps = eps

    def forward(self, x):
        return x / (torch.norm(x, p=2, dim=1, keepdim=True) + self.eps).expand_as(x)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'eps=' + str(self.eps) + ')'

class StdConv2d(nn.Conv2d):
    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-10)
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride, padding=0, bias=bias)


class PreActBottleneck(nn.Module):
  """
  Follows the implementation of "Identity Mappings in Deep Residual Networks" here:
  https://github.com/KaimingHe/resnet-1k-layers/blob/master/resnet-pre-act.lua

  Except it puts the stride on 3x3 conv when available.
  """

  def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cin)
        self.conv1 = conv1x1(cin, cmid)
        self.gn2 = nn.GroupNorm(32, cmid)
        # Original ResNetv2 has it on conv1!!
        self.conv2 = conv3x3(cmid, cmid, stride)
        self.gn3 = nn.GroupNorm(32, cmid)
        self.conv3 = conv1x1(cmid, cout)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride)

  def forward(self, x):
      # Conv'ed branch
      out = self.relu(self.gn1(x))

      # Residual branch
      residual = x
      if hasattr(self, 'downsample'):
          residual = self.downsample(out)

      # The first block has already applied pre-act before splitting, see Appendix.
      out = self.conv1(out)
      out = self.conv2(self.relu(self.gn2(out)))
      out = self.conv3(self.relu(self.gn3(out)))

      return out + residual


class ResNetV2(nn.Module):
    BLOCK_UNITS = {
        'r50': [3, 4, 6, 3],
        'r101': [3, 4, 23, 3],
        'r152': [3, 8, 36, 3],
    }

    def __init__(self, block_units, width_factor, head_size=21843, zero_head=False):
        super().__init__()
        wf = width_factor  # shortcut 'cause we'll use it a lot.

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, 64*wf, kernel_size=7, stride=2, padding=3, bias=False)),
            ('padp', nn.ConstantPad2d(1, 0)),
            ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0)),
            # The following is subtly not the same!
            #('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=64*wf, cout=256*wf, cmid=64*wf))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=256*wf, cout=256*wf, cmid=64*wf))
                for i in range(2, block_units[0] + 1)],
            ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=256*wf, cout=512*wf, cmid=128*wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=512*wf, cout=512 *
                                                wf, cmid=128*wf)) for i in range(2, block_units[1] + 1)],
            ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=512*wf, cout=1024*wf, cmid=256*wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=1024*wf, cout=1024 *
                                                wf, cmid=256*wf)) for i in range(2, block_units[2] + 1)],
            ))),
            ('block4', nn.Sequential(OrderedDict(
                [('unit01', PreActBottleneck(cin=1024*wf, cout=2048*wf, cmid=512*wf, stride=2))] +
                [(f'unit{i:02d}', PreActBottleneck(cin=2048*wf, cout=2048 *
                                                wf, cmid=512*wf)) for i in range(2, block_units[3] + 1)],
            ))),
        ]))

        self.zero_head = zero_head
        self.head = nn.Sequential(OrderedDict([
            ('gn', nn.GroupNorm(32, 2048*wf)),
            ('relu', nn.ReLU(inplace=True)),
            ('avg', nn.AdaptiveAvgPool2d(output_size=1)),
            ('conv', nn.Conv2d(2048*wf, head_size, kernel_size=1, bias=True)),
        ]))


    def forward(self, x):
        return self.body(self.root(x))
        # x = self.head(self.body(self.root(x)))
        # assert x.shape[-2:] == (1, 1)  # We should have no spatial shape left.
        # return x[..., 0, 0]


class resnet50bt(nn.Module):
    def __init__(self,model_path, feature_name):
        super(resnet50bt, self).__init__()
        res50bt = ResNetV2(ResNetV2.BLOCK_UNITS['r50'], width_factor=1, head_size=3000)
        res50bt = torch.nn.DataParallel(res50bt)
        checkpoint=torch.load(model_path, map_location="cpu")
        res50bt.load_state_dict(checkpoint['model'])
        self.norm=L2N()
        self.backbone = res50bt
        self.feature_name=feature_name
        if self.feature_name=='rmac':
            self.rmac=Rmac_Pooling()
        if self.feature_name=='ramac':
            self.ramac=Ramac_Pooling()
        if self.feature_name=='Grmac':
            self.Grmac=Grmac_Pooling()
        if self.feature_name=='Mac':
            self.Mac=Mac_Pooling()


    def forward(self,data):
        feature=self.backbone(data)
        if self.feature_name=='rmac':
            feature=self.rmac(feature)
        if self.feature_name=='ramac':
            feature=self.ramac(feature)
        if self.feature_name=='Grmac':
            feature=self.Grmac(feature)
        if self.feature_name=='Mac':
            feature=self.Mac(feature)
        feature=self.norm(feature)
        return feature

class resnet101bt(nn.Module):
    def __init__(self,model_path, feature_name):
        super(resnet101bt, self).__init__()
        res101bt = ResNetV2(ResNetV2.BLOCK_UNITS['r101'], width_factor=1, head_size=3000)
        res101bt = torch.nn.DataParallel(res101bt)
        checkpoint=torch.load(model_path, map_location="cpu")
        res101bt.load_state_dict(checkpoint['model'])
        self.norm=L2N()
        self.backbone = res101bt
        self.feature_name=feature_name
        if self.feature_name=='rmac':
            self.rmac=Rmac_Pooling()
        if self.feature_name=='ramac':
            self.ramac=Ramac_Pooling()
        if self.feature_name=='Grmac':
            self.Grmac=Grmac_Pooling()
        if self.feature_name=='Mac':
            self.Mac=Mac_Pooling()


    def forward(self,data):
        feature=self.backbone(data)
        if self.feature_name=='rmac':
            feature=self.rmac(feature)
        if self.feature_name=='ramac':
            feature=self.ramac(feature)
        if self.feature_name=='Grmac':
            feature=self.Grmac(feature)
        if self.feature_name=='Mac':
            feature=self.Mac(feature)
        feature=self.norm(feature)
        return feature