# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import clip
from domainbed.lib import wide_resnet

model_select = {
    'RN50': 'resnet50',
    'ViT': 'vit_b_16',
    'RN50_CLIP': 'RN50',
    'ViT_CLIP': 'ViT-B/16',
}
class Identity(nn.Module):
    """An identity layer"""

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SqueezeLastTwo(nn.Module):
    """
    A module which squeezes the last two dimensions,
    ordinary squeeze can be a problem for batch size 1
    """

    def __init__(self):
        super(SqueezeLastTwo, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], x.shape[1])


class MLP(nn.Module):
    """Just  an MLP"""

    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams["mlp_width"])
        self.dropout = nn.Dropout(hparams["mlp_dropout"])
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(hparams["mlp_width"], hparams["mlp_width"])
                for _ in range(hparams["mlp_depth"] - 2)
            ]
        )
        self.output = nn.Linear(hparams["mlp_width"], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class CLIP(nn.Module):
    def __init__(self, input_shape, hparams, class_token=None, network=None) -> None:
        super(CLIP, self).__init__()

        self.hparams = hparams
        CLIP_Net, self.preprocess = clip.load(model_select[hparams['backbone'] + '_CLIP'], device="cuda", jit=False)

        if hparams["algorithm"] != 'ERM':
            self.texts = self.to_texts_features(CLIP_Net, class_token)

        if hparams["CLIP"]:
            self.network = CLIP_Net
            self.logit_scale = self.network.logit_scale
        else: 
            del CLIP_Net
            torch.cuda.empty_cache()
            
            self.network = torchvision.models.resnet50(pretrained=hparams["pretrained"])
            self.network.fc = Identity()
            # self.extension = nn.Linear(in_features=2048, out_features=self.texts.size(-1))
            dropout = nn.Dropout(0.25)
            self.extension = nn.Sequential(
            nn.Linear(2048,
                      2048),
            dropout,
            nn.Linear(2048,
                      self.texts.size(-1)),
        )

            self.texts = self.texts.float()
            self.dropout = nn.Dropout(hparams["resnet_dropout"])
        
        self.freeze_bn()

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        if self.hparams["CLIP"]:
            image_features = self.network.encode_image(x)
        else:
            image_features = self.dropout(self.network(x))
            image_features = self.extension(image_features)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # text_features = self.texts

        # cosine similarity as logits
        # logit_scale = self.logit_scale.exp()
        logits_per_image = image_features @ self.texts.t()

        return logits_per_image, image_features

    def forward_features(self, x):
        """Encode x into a feature vector of size n_outputs."""
        if self.hparams["CLIP"]:
            image_features = self.network.encode_image(x)
        else:
            image_features = self.network.forward_features(x)
            image_features = self.network.global_pool(image_features)
            image_features = self.extension(image_features)
        return image_features

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        if self.hparams["freeze_bn"] is False:
            return

        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def to_texts_features(self, CLIP_Net, class_token):
        text_features = CLIP_Net.encode_text(class_token)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        return text_features.detach()

class ResNet(torch.nn.Module):
    """ResNet with the softmax chopped off and the batchnorm frozen"""

    def __init__(self, input_shape, hparams, network=None):
        super(ResNet, self).__init__()
        if hparams["resnet18"]:
            if network is None:
                self.network = torchvision.models.resnet18(pretrained=hparams["pretrained"])
            
            self.n_outputs = 512
        else:
            if network is None:
                self.network = torchvision.models.resnet50(pretrained=hparams["pretrained"])
            self.n_outputs = 2048

        # adapt number of channels
        nc = input_shape[0]
        if nc != 3:
            tmp = self.network.conv1.weight.data.clone()

            self.network.conv1 = nn.Conv2d(
                nc, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )

            for i in range(nc):
                self.network.conv1.weight.data[:, i, :, :] = tmp[:, i % 3, :, :]

        # save memory
        # del self.network.fc
        self.network.fc = Identity()

        self.hparams = hparams
        self.dropout = nn.Dropout(hparams["resnet_dropout"])
        self.freeze_bn()

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        if self.hparams["freeze_bn"] is False:
            return

        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """

    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.squeezeLastTwo = SqueezeLastTwo()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = self.squeezeLastTwo(x)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    if len(input_shape) == 1:
        return MLP(input_shape[0], 128, hparams)
    elif input_shape[1:3] == (28, 28):
        return MNIST_CNN(input_shape)
    elif input_shape[1:3] == (32, 32):
        return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.0)
    elif hparams["CLIP"]:
        return CLIP(input_shape, hparams)
    elif input_shape[1:3] == (224, 224):
        return ResNet(input_shape, hparams)
    else:
        raise NotImplementedError(f"Input shape {input_shape} is not supported")

def fea_proj(hparams, out_dim):
    if hparams['dataset'] == "DomainNet":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 1024
        hparams['dim'] = 512
        hparams['out_dim'] = out_dim
        fea_proj = nn.Sequential(
            nn.Linear(hparams['hidden_size'],
                      hparams['dim']),
            dropout,
            nn.Linear(hparams['dim'],
                      hparams['out_dim']),
        )
    elif hparams['dataset'] == "TerraIncognita":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 1024
        hparams['dim'] = 128
        hparams['out_dim'] = out_dim
        fea_proj = nn.Sequential(
            nn.Linear(hparams['hidden_size'],
                      hparams['out_dim']),
        )
    elif hparams['dataset'] == "OfficeHome":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 1024
        hparams['out_dim'] = out_dim
        fea_proj = nn.Sequential(
            nn.Linear(hparams['hidden_size'],
                      hparams['hidden_size']),
            dropout,
            nn.Linear(hparams['hidden_size'],
                      hparams['out_dim']),
        )
    elif hparams['dataset'] == "VLCS":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 1024
        hparams['dim'] = 128
        hparams['out_dim'] = out_dim
        fea_proj = nn.Sequential(
            nn.Linear(hparams['hidden_size'],
                      hparams['dim']),
            dropout,
            nn.Linear(hparams['dim'],
                      hparams['out_dim']),
        )
    elif hparams['dataset'] == "PACS":
        dropout = nn.Dropout(0.25)
        hparams['hidden_size'] = 1024
        hparams['dim'] = 128
        hparams['out_dim'] = out_dim
        fea_proj = nn.Sequential(
            nn.Linear(hparams['hidden_size'],
                      hparams['dim']),
            dropout,
            nn.Linear(hparams['dim'],
                      hparams['out_dim']),
        )
    else:
        pass
    
    return fea_proj