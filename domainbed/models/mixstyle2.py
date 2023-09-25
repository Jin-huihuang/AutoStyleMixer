"""
https://github.com/KaiyangZhou/mixstyle-release/blob/master/imcls/models/mixstyle.py
"""
import math
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
        if self.hparams['method'] == 'F':
            self.register_buffer("style", None)
        else:
            self.register_buffer("style", torch.zeros(2, self.domain_n, 1, num_features, 1, 1)) # dim 2 denote (mean, var)
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
                self.lmda = torch.nn.Parameter(torch.zeros(num_features, 2))
            else:
                self.lmda = torch.nn.Parameter(torch.zeros(1, 2))
            self.softmax = nn.Softmax(dim=-1)
        if self.hparams["AdaptiveAug"]:
            self.softmax = nn.Softmax(dim=-1)
            if self.hparams['fb']:
                self.lmda2 = torch.nn.Parameter(torch.zeros(num_features, 2))
            else:
                self.lmda2 = torch.nn.Parameter(torch.zeros(1, 2))

    def __repr__(self):
        return f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps})"

    def Multi_test(self, x, multi=False):
        B, C, H, W = x.shape
        if self.hparams['GB'] == 0:
            lmda = self.beta.sample((B, 1, 1, 1))
            lmda = lmda.to(x.device)
        elif self.hparams['GB'] == 1: 
            lmda = F.softmax(self.lmda * self.T).permute(1,0).view(2, -1, 1, 1)
            lmda = (lmda >= 0.5).float()
        if self.hparams['AdaptiveAug']:
            lmda2 = self.softmax(self.lmda*self.T)[:,0].view(-1, C, 1, 1)

        mu0 = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig0 = (var + self.eps).sqrt()
        if multi:
            domain_n = self._buffers['style'].size(1)
            # mu0, sig0 = mu.mean(dim=0, keepdim=True), sig.mean(dim=0, keepdim=True)
            x_normed = (x - mu0) / sig0
            multi_x = x
            for n in range(domain_n):
                mu = self._buffers['style'][0, n]
                sig = self._buffers['style'][1, n]
                if self.hparams['AdaptiveAug']:
                    mu = mu0 * (lmda[0:1] * lmda2 + lmda[1:2]) + mu * lmda[0:1] * (1 - lmda2)
                    sig = sig0 * (lmda[0:1] * lmda2 + lmda[1:2]) + sig * lmda[0:1] * (1 - lmda2)
                elif self.hparams['GB']:
                    mu = mu0 * lmda[0:1] + mu * lmda[1:2]
                    sig = sig0 * lmda[0:1] + sig * lmda[1:2]
                x_mix = x_normed * sig + mu
                multi_x = torch.concat([multi_x, x_mix], dim=0)
        else:
            domain_n = self._buffers['style'].size(1)
            test_size = x.size(0) // (domain_n + 1)
            # mu, sig = mu.view(domain_n + 1, test_size, -1, 1, 1).mean(dim=1, keepdim=True), sig.view(domain_n + 1, test_size, -1, 1, 1).mean(dim=1, keepdim=True)
            # mu, sig = mu.repeat(1, test_size, 1, 1, 1).view(x.size(0), -1, 1, 1), sig.repeat(1, test_size, 1, 1, 1).view(x.size(0), -1, 1, 1)
            x_normed = (x - mu0) / sig0
            x_normed[0:test_size] = x[0:test_size]
            x = x_normed
            for n in range(domain_n):
                mu = self._buffers['style'][0, n]
                sig = self._buffers['style'][1, n]
                index_s = (n+1) * test_size
                index_e = (n+2) * test_size
                if self.hparams['AdaptiveAug']:
                    mu = mu0[index_s:index_e] * (lmda[0:1] * lmda2 + lmda[1:2]) + mu * lmda[0:1] * (1 - lmda2)
                    sig = sig0[index_s:index_e] * (lmda[0:1] * lmda2 + lmda[1:2]) + sig * lmda[0:1] * (1 - lmda2)
                elif self.hparams['GB']:
                    mu = mu0[index_s:index_e] * lmda[0:1] + mu * lmda[1:2]
                    sig = sig0[index_s:index_e] * lmda[0:1] + sig * lmda[1:2]
                x[index_s:index_e] = x[index_s:index_e] * sig + mu
            multi_x = x
        return multi_x
    
    def forward(self, x, activated=False, multi=False):
        """
        For the input x, the first half comes from one domain,
        while the second half comes from the other domain.
        """
        if not self.training:
            if not self.hparams['Multi_test']:
                return x
            else:
                return self.Multi_test(x, multi)
        content, style, old_style = self.decouple(x)
        
        if self.hparams['nvs1']:
            if not activated or not self.MT:
                if self._buffers['style'] is None:  # Check if it's still uninitialized
                    self._buffers['style'] = old_style
                elif not self.hparams['EMA']:
                    self._buffers['style'] = old_style
                else:
                    self._buffers['style'] = self._buffers['style'] * self.momentum + old_style * (1 - self.momentum)
                if self.MT:
                    return x
        mix_style = self.mix_style(content, style)

        return self.couple(content, mix_style)
    
    def decouple(self, x):
        if self.hparams['method'] == 'F':
            content, style, old_style = self.Fourier(x)
        else:
            content, style, old_style = self.Norm(x)
        return content, style, old_style

    def mix_style(self, content, style):
        B, C, H, W = content.shape
        # mix_style, shuffle
        ori_perm = torch.arange(self.domain_n)
        while True:
            perm = torch.randperm(ori_perm.size(0))
            if torch.all(perm != ori_perm):
                break

        if self.hparams['random']:
            lmda = self.beta.sample((B, 1, 1, 1))
            lmda = lmda.to(content.device)
        # 1.Gumbel-Softmax
        elif self.hparams['GB'] == 1: 
            lmda = F.gumbel_softmax(self.lmda, 1/self.T, hard=True).view(2, -1, 1, 1)
        else:
            lmda = self.softmax(self.lmda*self.T)[:,0].view(-1, C, 1, 1)

        if self.hparams['nvs1']:
            if self.hparams['method'] == "F":
                style2 = self._buffers['style'][perm].repeat(1, B//self.domain_n, 1, 1, 1).view(B, C, H, W)
            else:
                style2 = [self._buffers['style'][0][perm].repeat(1, B//self.domain_n, 1, 1, 1).view(B, C, 1, 1), self._buffers['style'][1][perm].repeat(1, B//self.domain_n, 1, 1, 1).view(B, C, 1, 1)]
        else:
            style2 = style.detach().view(self.domain_n, B//self.domain_n, C, H, W)[perm].view(B, C, H, W)
        style = torch.stack(style) if isinstance(style, list) else style
        style2 = torch.stack(style2) if isinstance(style2, list) else style2

        if self.hparams['AdaptiveAug']:
            lmda2 = self.softmax(self.lmda2 * self.T).view(-1, C, 1, 1)
            mix_style = style * (lmda[0:1] * lmda2[0:1] + lmda[1:2]) + style2 * lmda[0:1] * lmda2[1:2]
        elif self.hparams['GB']:
            mix_style = style * lmda[0:1] + style2 * lmda[1:2]
        else:
            mix_style = style * lmda + style2 * (1 - lmda)
        return mix_style
    
    def couple(self, content, style):
        if self.hparams['method'] == 'F':
            return self.compose(content, style)
        else:
            return content * style[1] + style[0]

    def Norm(self, x):
        B, C, H, W = x.shape
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()

        new_statistics = torch.stack([mu, sig], dim=0).view(2, self.domain_n, B//self.domain_n, C, 1, 1).mean(dim=2, keepdim=True) # (sta,D,B/D,C,1,1) --> (sta,D,1,C,1,1)
        if self.hparams['bn']:
            mu, sig = new_statistics[0].repeat(1, B//self.domain_n, 1, 1, 1).view(B, C, 1, 1), new_statistics[1].repeat(1, B//self.domain_n, 1, 1, 1).view(B, C, 1, 1)
        x_normed = (x - mu) / sig
        return x_normed, [mu, sig], new_statistics

    def Fourier(self, x):
        B, C, H, W = x.shape
        fft_pha, fft_amp = self.decompose(x)
        old_amp = fft_amp.view(self.domain_n, B//self.domain_n, C, H, W).mean(dim=1, keepdim=True)
        return fft_pha, fft_amp, old_amp

    def decompose(self, x):
        fft_im = torch.view_as_real(torch.fft.fft2(x, norm='backward'))
        fft_amp = fft_im.pow(2).sum(dim=-1, keepdim=False)
        fft_amp = torch.sqrt(self.replace_denormals(fft_amp))
        fft_pha = torch.atan2(fft_im[..., 1], self.replace_denormals(fft_im[..., 0]))
        return fft_pha.detach(), fft_amp.detach()

    def compose(self, phase, amp):
        x = torch.stack([torch.cos(phase) * amp, torch.sin(phase) * amp], dim=-1) 
        x = x / math.sqrt(x.shape[2] * x.shape[3])
        x = torch.view_as_complex(x)
        return torch.fft.irfft2(x, s=x.shape[2:], norm='ortho')
    
    def replace_denormals(self, x, threshold=1e-5):
        y = x.clone()
        y[(x < threshold)&(x > -1.0 * threshold)] = threshold
        return y