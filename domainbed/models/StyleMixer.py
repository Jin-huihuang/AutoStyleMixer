import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
    
class StyleMixer(nn.Module):
    """
    StyleMixer: adaptively mixes the style with Normlization or Fourier Transform.
    """

    def __init__(self, initial_value=0.5, T=0.3, eps=1e-6, lmda=None, num_features=None, domain_n=None, momentum=0.9, **kwargs):
        super().__init__()
        self.eps = eps
        self.T = T
        self.lmda = lmda
        self.momentum = momentum
        self._activated = True
        self.hparams = kwargs['hparams']
        self.MT = self.hparams['MT']

        self.domain_n = domain_n - 1 # leave-one-out
        self.beta = torch.distributions.Beta(0.1, 0.1)
        if self.hparams['method'] == 'F':
            self.register_buffer('statistics', None)
        else:
            self.register_buffer('statistics', torch.zeros(2, self.domain_n, 1, num_features, 1, 1)) # dim 2 denote (mean, var)
        
        if not self.hparams['random'] and self.hparams["AdaptiveAug"]:
            self.lmda = torch.nn.Parameter(torch.zeros(num_features, 2))
            self.softmax = nn.Softmax(dim=-1)
        if self.hparams["AdaptiveAug"]:
            self.softmax = nn.Softmax(dim=-1)
            self.lmda2 = torch.nn.Parameter(torch.zeros(num_features, 2))

    def __repr__(self):
        return f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps})"
    
    def forward(self, x, activated=False, multi=False):
        """
        For the input x, the first half comes from one domain,
        while the second half comes from the other domain.
        """
        if not self.training:
            return x
        content, style, old_style = self.decouple(x)

        if self.hparams['nvs1']:
            if not activated or not self.MT:
                if self._buffers['statistics'] is None:  # Check if it's still uninitialized
                    self._buffers['statistics'] = old_style
                elif not self.hparams['EMA']:
                    self._buffers['statistics'] = old_style
                else:
                    self._buffers['statistics'] = self._buffers['statistics'] * self.momentum + old_style * (1 - self.momentum)
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
        while True: # random select other domain
            perm = torch.randperm(ori_perm.size(0))
            if torch.all(perm != ori_perm):
                break

        if self.hparams['random'] and not self.hparams['AdaptiveAug']: # MixStyle
            lmda = self.beta.sample((B, 1, 1, 1))
            lmda = lmda.to(content.device)
        # 1.Gumbel-Softmax
        elif self.hparams['AdaptiveP']: 
            lmda = F.gumbel_softmax(self.lmda, 1/self.T, hard=True).view(2, -1, 1, 1)
        else:
            lmda = self.softmax(self.lmda*self.T)[:,0].view(-1, C, 1, 1)

        if self.hparams['nvs1']:
            if self.hparams['method'] == "F":
                style2 = self._buffers['statistics'][perm].repeat(1, B//self.domain_n, 1, 1, 1).view(B, C, H, W)
            else:
                style2 = [self._buffers['statistics'][0][perm].repeat(1, B//self.domain_n, 1, 1, 1).view(B, C, 1, 1), self._buffers['statistics'][1][perm].repeat(1, B//self.domain_n, 1, 1, 1).view(B, C, 1, 1)]
        else:
            if self.hparams['method'] == 'F':
                style2 = style.detach().view(self.domain_n, B//self.domain_n, C, H, W)[perm].view(B, C, H, W)
            else:
                style2 = [style[0].detach().view(self.domain_n, B//self.domain_n, C, 1, 1)[perm].view(B, C, 1, 1), style[1].detach().view(self.domain_n, B//self.domain_n, C, 1, 1)[perm].view(B, C, 1, 1)]
        style = torch.stack(style) if isinstance(style, list) else style
        style2 = torch.stack(style2) if isinstance(style2, list) else style2

        if self.hparams['AdaptiveAug']:
            assert self.hparams['AdaptiveP'] or self.hparams['AdaptiveW'], "At least one of AdaptiveP or AdaptiveW is True"
            if self.hparams['AdaptiveP'] and self.hparams['AdaptiveW']:
                lmda2 = self.softmax(self.lmda2 * self.T).view(-1, C, 1, 1)
                mix_style = style * (lmda[0:1] * lmda2[0:1] + lmda[1:2]) + style2 * lmda[0:1] * lmda2[1:2]
            elif not self.hparams['AdaptiveP'] and self.hparams['AdaptiveW']:
                lmda2 = self.softmax(self.lmda2 * self.T).view(-1, C, 1, 1)
                if random.random() > 0.5:
                    return style
                mix_style = style * lmda2[0:1] + style2 * lmda2[1:2]
            elif self.hparams['AdaptiveP'] and not self.hparams['AdaptiveW']:
                lmda2 = self.beta.sample((B, 1, 1, 1)).to(content.device)
                mix_style = style * (lmda[0:1] * lmda2 + lmda[1:2]) + style2 * lmda[0:1] * (1 - lmda2)
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