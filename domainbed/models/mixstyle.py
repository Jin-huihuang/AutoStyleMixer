"""
https://github.com/KaiyangZhou/mixstyle-release/blob/master/imcls/models/mixstyle.py
"""
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.3, eps=1e-6):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha

        print("* MixStyle params")
        print(f"- p: {p}")
        print(f"- alpha: {alpha}")

    def __repr__(self):
        return f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps})"

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x - mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.randperm(B)
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix
        
class MixStyle2(nn.Module):
    """MixStyle (w/ domain prior).
    The input should contain two equal-sized mini-batches from two distinct domains.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, initial_value=0.5, T=0.3, eps=1e-6, lmda=None, num_features=None, domain_n=None, momentum=0.9, **kwargs):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
        """
        super().__init__()
        self.eps = eps
        self.T = T
        self.lmda = lmda
        self.momentum = momentum
        self._activated = True
        self.hparams = kwargs['hparams']
        self.MT = self.hparams['MT']

        self.domain_n = domain_n - 1 # leave-one-out
        self.register_buffer("statistics", None) # dim 2 denote (mean, var)
        if self.hparams['GB'] == 2:
            if self.hparams['fb']:
                self.variation = nn.Sequential(
            nn.Linear(num_features, num_features, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, num_features, bias=False),
            nn.Sigmoid()
            )
            else:
                self.variation = nn.Sequential(
                nn.Linear(num_features, num_features, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(num_features, 1, bias=False),
                nn.Sigmoid()
                )
        elif self.hparams['random']:
            self.beta = torch.distributions.Beta(0.1, 0.1)
        else:
            if self.hparams['fb']:
                self.lmda = torch.nn.Parameter(torch.full((num_features, 2), initial_value))
            else:
                self.lmda = torch.nn.Parameter(torch.full((1, 2), initial_value))
            self.softmax = nn.Softmax(dim=-1)
            

    def __repr__(self):
        return f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps})"

    def forward(self, x, activated=False):
        """
        For the input x, the first half comes from one domain,
        while the second half comes from the other domain.
        """
        if not self.training:
            return x
        
        B, C, H, W = x.shape

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        
        new_statistics = torch.stack([mu, sig], dim=0).view(2, self.domain_n, B//self.domain_n, C, 1, 1).mean(dim=2, keepdim=True) # (sta,D,B/D,C,1,1)
        if self.hparams['bn']:
            mu, sig = new_statistics[0].repeat(1, B//self.domain_n, 1, 1, 1).view(B, C, 1, 1), new_statistics[1].repeat(1, B//self.domain_n, 1, 1, 1).view(B, C, 1, 1)
        
        # EMA statistics
        if not activated or not self.MT:
            if self._buffers['statistics'] is not None:
                self._buffers['statistics'] = self._buffers['statistics']*self.momentum + new_statistics*(1-self.momentum)
            else:
                self._buffers['statistics'] = new_statistics
            if self.MT:    
                return x
        # 2.reinforce
            
        # mix_style, shuffle
        perm = torch.randperm(self.domain_n)
        mu2, sig2 = self._buffers['statistics'][0][perm].repeat(1, B//self.domain_n, 1, 1, 1).view(B, C, 1, 1), self._buffers['statistics'][1][perm].repeat(1, B//self.domain_n, 1, 1, 1).view(B, C, 1, 1)
        x_normed = (x - mu) / sig
        if self.hparams['random']:
            lmda = self.beta.sample((B, 1, 1, 1))
            lmda = lmda.to(x.device)
        # 1.Gumbel-Softmax
        elif self.hparams['GB'] == 1: 
            lmda = F.gumbel_softmax(self.lmda, 1/self.T, hard=True).view(2, -1, 1, 1)
        elif self.hparams['GB'] == 2:
            if self.hparams['detach']:
                x_avg = x.detach().mean(dim=(0,2,3))
            else:
                x_avg = x.mean(dim=(0,2,3))
            x_variation = self.variation(x_avg)
            lmda = F.gumbel_softmax(torch.cat((x_variation,1-x_variation)), 1/self.T, hard=True).view(2, -1, 1, 1)
        else:
            lmda = self.softmax(self.lmda*self.T)[:,0].view(-1, C, 1, 1)
        
        # if self.hparams['c_dropout']:
        #     lmda = lmda.expand(mu.size())
        #     mask = lmda.new_empty(lmda.shape).bernoulli_(1 - self.hparams['c_drop_prob'])
        #     lmda = 1 - lmda * mask
        if self.hparams['GB']:
            mu_mix = mu * lmda[0:1] + mu2 * lmda[1:2]
            sig_mix = sig * lmda[0:1] + sig2 * lmda[1:2]
        else:
            mu_mix = mu * lmda + mu2 * (1 - lmda)
            sig_mix = sig * lmda + sig2 * (1 - lmda)

        return x_normed * sig_mix + mu_mix
