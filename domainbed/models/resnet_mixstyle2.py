"""MixStyle w/ domain label
https://github.com/KaiyangZhou/mixstyle-release/blob/master/imcls/models/resnet_mixstyle2.py
"""
import random
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .mixstyle import MixStyle2 as MixStyle

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self, block, layers, mixstyle_layers=[], mixstyle_p=0.5, mixstyle_alpha=0.3, **kwargs
    ):
        self.inplanes = 64
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.domain = None

        self.mixstyle = None
        self._activated = True
        self.eps = 1e-6
        self.mu = []
        self.var = []
        self.style_mus = []
        self.style_vars = []
        if mixstyle_layers:
            self.mixstyle = MixStyle(p=mixstyle_p, T=kwargs['hparams']['Mix_T'], lmda=kwargs["hparams"]['lmda'])
            for layer_name in mixstyle_layers:
                assert layer_name in ["conv1", "conv2_x", "conv3_x", "conv4_x", "conv5_x"]
                mean_buffer = None
                var_buffer = None
                self.register_buffer('style_mean_' + layer_name, mean_buffer) # (N, C) N domains, C channels
                self.style_mus.append(self._buffers['style_mean_' + layer_name])
                self.register_buffer('style_var_' + layer_name, var_buffer)
                self.style_vars.append(self._buffers['style_var_' + layer_name])

            print("Insert MixStyle after the following layers: {}".format(mixstyle_layers))
        self.mixstyle_layers = mixstyle_layers
        
        self._out_features = 512 * block.expansion
        self.fc = nn.Identity()  # for DomainBed compatibility

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def compute_style(self, x):
        mu = x.mean(dim=[2, 3])
        sig = x.std(dim=[2, 3])
        return torch.cat([mu, sig], 1)
    
    def set_activation_status(self, status=True, domain=None):
        self._activated = status
        self.domain = domain

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        mu = []
        var = []
        x = self.layer1(x)
        l = 0
        if "conv2_x" in self.mixstyle_layers:
            if self._activated:
                x = self.mixstyle(x, self.domain, self.style_mus, self.style_vars, layer=l)
                l+=1
            else:
                mu.append(x.mean(dim=[0, 2, 3]).detach())
                var.append((x.var(dim=[0, 2, 3]) + self.eps).sqrt().detach())

        x = self.layer2(x)
        if "conv3_x" in self.mixstyle_layers:
            if self._activated:
                x = self.mixstyle(x, self.domain, self.style_mus, self.style_vars, layer=l)
                l+=1
            else:
                mu.append(x.mean(dim=[0, 2, 3]).detach())
                var.append((x.var(dim=[0, 2, 3]) + self.eps).sqrt().detach())

        x = self.layer3(x)
        if "conv4_x" in self.mixstyle_layers:
            if self._activated:
                x = self.mixstyle(x, self.domain, self.style_mus, self.style_vars, layer=l)
                l+=1
            else:
                mu.append(x.mean(dim=[0, 2, 3]).detach())
                var.append((x.var(dim=[0, 2, 3]) + self.eps).sqrt().detach())

        x = self.layer4(x)
        if "conv5_x" in self.mixstyle_layers:
            if self._activated:
                x = self.mixstyle(x, self.domain, self.style_mus, self.style_vars, layer=l)
                l+=1
            else:
                mu.append(x.mean(dim=[0, 2, 3]).detach())
                var.append((x.var(dim=[0, 2, 3]) + self.eps).sqrt().detach())
        if not self._activated:
            self.mu.append(mu)
            self.var.append(var)
        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)
    
    def update_buffers(self, momentum=0.9):
        if self.style_mus[0]:
            for l1, l2 in zip(self.style_mus, self.mu): # domain
                for i, (x, y) in enumerate(zip(l1, l2)): # layer
                    l1[i] = momentum * x + (1. - momentum) * y
            for l1, l2 in zip(self.style_vars, self.var): # domain
                for i, (x, y) in enumerate(zip(l1, l2)): # layer
                    l1[i] = momentum * x + (1. - momentum) * y
        else:
            self.style_mus = self.mu
            self.style_vars = self.var
        self.mu = []
        self.var = []


class ResNet2(nn.Module):
    def __init__(
        self, block, layers, mixstyle_layers=[], mixstyle_p=0.5, mixstyle_alpha=0.3, **kwargs
    ):
        num_features = dict()
        self.inplanes = 64
        self.hparams = kwargs["hparams"]
        super().__init__()

        # backbone network
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        num_features["conv1"] = 64
        self.layer1 = self._make_layer(block, 64, layers[0])
        num_features["conv2_x"] = self.inplanes
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        num_features["conv3_x"] = self.inplanes
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        num_features["conv4_x"] = self.inplanes
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        num_features["conv5_x"] = self.inplanes

        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.mixstyle_layers = mixstyle_layers
        self.stymix = nn.ModuleDict([])
        for layer_name in mixstyle_layers:
            self.stymix[layer_name] = MixStyle(
                initial_value=self.hparams['initial_value'], 
                T=self.hparams['Mix_T'],
                num_features=num_features[layer_name],
                domain_n=self.hparams['domain_num'], 
                momentum=self.hparams["momentum_style"],
                hparams=self.hparams
                )

        self._activated = True

        self._out_features = 512 * block.expansion
        self.fc = nn.Identity()  # for DomainBed compatibility

        self._init_params()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def compute_style(self, x):
        mu = x.mean(dim=[2, 3])
        sig = x.std(dim=[2, 3])
        return torch.cat([mu, sig], 1)
    
    def set_activation_status(self, status=True, domain=None):
        self._activated = status
        self.domain = domain

    def featuremaps(self, x):
        if self.training:
            multi_flag = False
        else:
            multi_flag = True
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # (B, C, H, W)
        if "conv1" in self.mixstyle_layers:
            if multi_flag:
                multi_flag = False
                multi = True
            else:
                multi = False
            x_tmp = self.stymix["conv1"](x, self._activated, multi)
            if self.hparams['resdual']:
                x = x.repeat(x_tmp.size(0)//x.size(0), 1, 1, 1)
                if self.hparams['div2']:
                    x = (x + x_tmp) / 2
                else:
                    x = x + x_tmp
            else:
                x = x_tmp

        x = self.layer1(x)
        if "conv2_x" in self.mixstyle_layers:
            if multi_flag:
                multi_flag = False
                multi = True
            else:
                multi = False
            x_tmp = self.stymix["conv2_x"](x, self._activated, multi)
            if self.hparams['resdual']:
                x = x.repeat(x_tmp.size(0)//x.size(0), 1, 1, 1)
                if self.hparams['div2']:
                    x = (x + x_tmp) / 2
                else:
                    x = x + x_tmp
            else:
                x = x_tmp

        x = self.layer2(x)
        if "conv3_x" in self.mixstyle_layers:
            if multi_flag:
                multi_flag = False
                multi = True
            else:
                multi = False
            x_tmp = self.stymix["conv3_x"](x, self._activated, multi)
            if self.hparams['resdual']:
                x = x.repeat(x_tmp.size(0)//x.size(0), 1, 1, 1)
                if self.hparams['div2']:
                    x = (x + x_tmp) / 2
                else:
                    x = x + x_tmp
            else:
                x = x_tmp

        x = self.layer3(x)
        if "conv4_x" in self.mixstyle_layers:
            if multi_flag:
                multi_flag = False
                multi = True
            else:
                multi = False
            x_tmp = self.stymix["conv4_x"](x, self._activated, multi)
            if self.hparams['resdual']:
                x = x.repeat(x_tmp.size(0)//x.size(0), 1, 1, 1)
                if self.hparams['div2']:
                    x = (x + x_tmp) / 2
                else:
                    x = x + x_tmp
            else:
                x = x_tmp

        x = self.layer4(x)
        if "conv5_x" in self.mixstyle_layers:
            if multi_flag:
                multi_flag = False
                multi = True
            else:
                multi = False
            x_tmp = self.stymix["conv5_x"](x, self._activated, multi)
            if self.hparams['resdual']:
                x = x.repeat(x_tmp.size(0)//x.size(0), 1, 1, 1)
                if self.hparams['div2']:
                    x = (x + x_tmp) / 2
                else:
                    x = x + x_tmp
            else:
                x = x_tmp

        return x

    def forward(self, x):
        f = self.featuremaps(x)
        v = self.global_avgpool(f)
        return v.view(v.size(0), -1)




def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


"""
Residual network configurations:
--
resnet18: block=BasicBlock, layers=[2, 2, 2, 2]
resnet34: block=BasicBlock, layers=[3, 4, 6, 3]
resnet50: block=Bottleneck, layers=[3, 4, 6, 3]
resnet101: block=Bottleneck, layers=[3, 4, 23, 3]
resnet152: block=Bottleneck, layers=[3, 8, 36, 3]
"""


def resnet18_mixstyle2_L234_p0d5_a0d1(pretrained=True, **kwargs):
    model = ResNet(
        block=BasicBlock,
        layers=[2, 2, 2, 2],
        mixstyle_layers=kwargs["hparams"]["mix_layers"],
        mixstyle_p=kwargs["hparams"]["mix_p"],
        mixstyle_alpha=kwargs["hparams"]["mix_a"],
        **kwargs
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet18"])

    return model


def resnet50_mixstyle2_L234_p0d5_a0d1(pretrained=True, **kwargs):
    model = ResNet2(
        block=Bottleneck,
        layers=[3, 4, 6, 3],
        mixstyle_layers=kwargs["hparams"]["mix_layers"],
        mixstyle_p=kwargs["hparams"]["mix_p"],
        mixstyle_alpha=kwargs["hparams"]["mix_a"],
        **kwargs
    )

    if pretrained:
        init_pretrained_weights(model, model_urls["resnet50"])

    return model
