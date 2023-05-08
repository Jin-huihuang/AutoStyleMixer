"""
https://github.com/KaiyangZhou/mixstyle-release/blob/master/imcls/models/mixstyle.py
"""
import random
import torch
import torch.nn as nn


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

    def __init__(self, p=0.5, alpha=0.3, eps=1e-6, lmda=None):
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
        self.lmda = lmda
        self._activated = True

        print("* MixStyle params")
        print(f"- p: {p}")
        print(f"- alpha: {alpha}")

    def __repr__(self):
        return f"MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps})"

    def forward(self, x, domain=None, mu_domains=None, var_domains=None, layer=0):
        """
        For the input x, the first half comes from one domain,
        while the second half comes from the other domain.
        """
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

        if self.lmda:
            lmda = torch.full((B, 1, 1, 1),self.lmda[layer])
        else:
            lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        perm = torch.arange(B - 1, -1, -1)  # inverse index
        perm_b, perm_a = perm.chunk(2)
        perm_b = perm_b[torch.randperm(B // 2)]
        perm_a = perm_a[torch.randperm(B // 2)]
        perm = torch.cat([perm_b, perm_a], 0)
        
        if mu_domains[0]:
            N = len(mu_domains)
            B, C, H, W = mu.shape
            mu_domain1 = mu_domains[domain][layer].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(16, 1, 1, 1)
            j = domain + 1 if domain < (N - 1) else 0
            mu_domain2 = mu_domains[j][layer].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(16, 1, 1, 1)
            mu2 = torch.cat([mu_domain2, mu_domain1])

            var_domain1 = var_domains[domain][layer].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(16, 1, 1, 1)
            var_domain2 = var_domains[j][layer].unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(16, 1, 1, 1)
            sig2 = torch.cat([var_domain2, var_domain1])

            mu_mix = mu * lmda + mu2 * (1 - lmda)
            sig_mix = sig * lmda + sig2 * (1 - lmda)

            return x_normed * sig_mix + mu_mix
        else:
            mu2, sig2 = mu[perm], sig[perm]
            mu_mix = mu * lmda + mu2 * (1 - lmda)
            sig_mix = sig * lmda + sig2 * (1 - lmda)

            return x_normed * sig_mix + mu_mix
