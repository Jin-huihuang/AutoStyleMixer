import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from domainbed.lib.fast_data_loader import FastDataLoader

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
Softmax = nn.Softmax(dim=1)

def accuracy_from_loader(algorithm, loader, weights, hparams, debug=False):
    correct = 0 
    correct_t = 0 # mean teacher
    total = 0
    losssum = 0.0
    losssum_t = 0.0
    weights_offset = 0

    algorithm.eval()

    for i, batch in enumerate(loader):
        x = batch["x"].to(device)
        y = batch["y"].to(device)

        with torch.no_grad():
            if hparams['MT']:
                logits, logits_t = algorithm.predict(x)
                loss = F.cross_entropy(logits, y).item()
                loss_t = F.cross_entropy(logits_t, y).item()
            else:
                logits = algorithm.predict(x)
                loss = F.cross_entropy(logits, y).item()

        B = len(x)
        losssum += loss * B
        if hparams['MT']:
            losssum_t += loss_t*B

        if weights is None:
            batch_weights = torch.ones(len(x))
        else:
            batch_weights = weights[weights_offset : weights_offset + len(x)]
            weights_offset += len(x)
        batch_weights = batch_weights.to(device)
        if logits.size(1) == 1:
            correct += (logits.gt(0).eq(y).float() * batch_weights).sum().item()
        else:
            correct += (logits.argmax(1).eq(y).float() * batch_weights).sum().item()
        if hparams['MT']:
            if logits.size(1) == 1:
                correct_t += (logits_t.gt(0).eq(y).float() * batch_weights).sum().item()
            else:
                correct_t += (logits_t.argmax(1).eq(y).float() * batch_weights).sum().item()
        total += batch_weights.sum().item()

        if debug:
            break

    algorithm.train()

    acc = correct / total
    acc_t = correct_t / total
    loss = losssum / total
    loss_t = losssum_t / total
    return acc, acc_t, loss, loss_t


def accuracy(algorithm, loader_kwargs, weights, hparams, **kwargs):
    if isinstance(loader_kwargs, dict):
        loader = FastDataLoader(**loader_kwargs)
    elif isinstance(loader_kwargs, FastDataLoader):
        loader = loader_kwargs
    else:
        raise ValueError(loader_kwargs)
    return accuracy_from_loader(algorithm, loader, weights, hparams,**kwargs)


class Evaluator:
    def __init__(
        self, test_envs, eval_meta, n_envs, logger, evalmode="fast", debug=False, target_env=None, hparams=None
    ):
        all_envs = list(range(n_envs))
        train_envs = sorted(set(all_envs) - set(test_envs))
        self.test_envs = test_envs
        self.train_envs = train_envs
        self.eval_meta = eval_meta
        self.n_envs = n_envs
        self.logger = logger
        self.evalmode = evalmode
        self.debug = debug
        self.hparams = hparams

        if target_env is not None:
            self.set_target_env(target_env)

    def set_target_env(self, target_env):
        """When len(test_envs) == 2, you can specify target env for computing exact test acc."""
        self.test_envs = [target_env]

    def evaluate(self, algorithm, A_method, ret_losses=False):
        n_train_envs = len(self.train_envs)
        n_test_envs = len(self.test_envs)
        assert n_test_envs == 1
        summaries = collections.defaultdict(float)
        # for key order
        summaries["test_in"] = 0.0
        summaries["test_out"] = 0.0
        summaries["train_in"] = 0.0
        summaries["train_out"] = 0.0

        summaries["test_inMT"] = 0.0
        summaries["test_outMT"] = 0.0
        accuracies = {}
        losses = {}

        # order: in_splits + out_splits.
        for name, loader_kwargs, weights in self.eval_meta:
            # env\d_[in|out]
            env_name, inout = name.split("_")
            env_num = int(env_name[3:])

            skip_eval = self.evalmode == "fast" and inout == "in" and env_num not in self.test_envs
            if skip_eval:
                continue

            is_test = env_num in self.test_envs
            acc, acc_t, loss, loss_t = accuracy(algorithm, loader_kwargs, weights, self.hparams, debug=self.debug)
            accuracies[name] = acc
            accuracies[name+'MT'] = acc_t
            losses[name] = loss
            losses[name+'MT'] = loss_t

            if env_num in self.train_envs:
                summaries["train_" + inout] += acc / n_train_envs
                if inout == "out":
                    summaries["tr_" + inout + "loss"] += loss / n_train_envs
            elif is_test:
                summaries["test_" + inout] += acc / n_test_envs
                summaries["test_" + inout + 'MT'] += acc_t / n_test_envs

        if ret_losses:
            return accuracies, summaries, losses
        else:
            return accuracies, summaries
