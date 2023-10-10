import argparse
from distutils.util import strtobool
from glob import glob
from re import split
import sys
from matplotlib.colors import ListedColormap
import torch
from domainbed.algorithms import MSMT2
import numpy as np
from domainbed.lib import misc
import numpy
from domainbed.datasets import datasets, split_dataset
import matplotlib.pyplot as pyplot
from tqdm import tqdm
from domainbed.datasets import transforms as DBT
from torch.utils.data.dataloader import DataLoader
from openTSNE import TSNE
from domainbed.lib.fast_data_loader import FastDataLoader
from matplotlib.gridspec import GridSpec

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def main():
    # network, output_dir
    parser = argparse.ArgumentParser(description="Domain generalization")
    parser.add_argument("--network", type=str, default="RN50")
    parser.add_argument("--output_dir",
                        default='/export/home/zhh/project/MHDG/train_output/PACS/230807_09-49-29_PACS_MSMT2_100_3e-05R1_M/checkpoints',
                        type=str, help="please add train_output pathdir to here, like 'train_output/OfficeHome/...'")
    parser.add_argument("--data_dir", default='/data', type=str)
    parser.add_argument("--mode", type=int, default=0, help='0:source_only, 1:target_only, 2:all')
    parser.add_argument("--method", type=str, default='N', help='0:source_only, 1:target_only, 2:all')
    parser.add_argument("--layer", type=list, default=[0,4], help='Visualize layers')
    args = parser.parse_args()
    
    output = split('[. /]', args.output_dir)
    output = [item for item in filter(lambda x:x != '', output)]
    dataset = vars(datasets)[output[6]+"_vis"](args.data_dir)

    for pth in glob(args.output_dir + "/*.pth"):
        class_names = dataset.datasets[0].classes
        domain_names = dataset.environments.copy()
        checkpoint = torch.load(pth)
        test_envs = checkpoint['test_envs']
        domain_names.pop(test_envs[0])
        checkpoint['model_hparams']['method'] = args.method
        model = MSMT2(dataset.input_shape, 7, len(dataset) - len(test_envs), checkpoint['model_hparams']).to(device)
        model.load_state_dict(checkpoint['model_dict'])
        model.eval()
        sum = None
        targets = None
        data, in_splits, out_splits = get_dataset(test_envs, dataset)
        n_envs = len(dataset)
        train_envs = sorted(set(range(n_envs)) - set(test_envs))
        iterator = misc.SplitIterator(test_envs)
        batch_sizes = np.full([n_envs], 32, dtype=int)
        batch_sizes[test_envs] = 0
        batch_sizes = batch_sizes.tolist()
        
        test_loaders = [
            FastDataLoader(
                dataset=env,
                batch_size=batch_size,
                num_workers=4,
            )
            for env, batch_size in iterator.train(zip(in_splits, batch_sizes))
        ]
        test_minibatches_iterator = zip(*test_loaders)
        min_step = min([len(loader) for loader in test_loaders])
        features_layer = []
        styles_layer = []
        targets = []
        domains = []
        for step in range(min_step):

            batches_dictlist = next(test_minibatches_iterator)
            sum = 0
            for batch in batches_dictlist:
                sum += batch['y'].size(0)
            if np.sum(batch_sizes) != sum:
                break
            for domain in range(len(batches_dictlist)):
                batches_dictlist[domain]['domain']=(torch.full(batches_dictlist[domain]['y'].size(), domain))
            # batches: {data_key: [env0_tensor, ...], ...}
            batches = misc.merge_dictlist(batches_dictlist)
            # to device
            batches = {key: [tensor.to(device) for tensor in tensorlist] for key, tensorlist in batches.items()}

            inputs = {**batches}
            with torch.no_grad():
                net = model.featurizer.network
                feature, style = net.embed(inputs['x'], mode=args.mode, layer=args.layer)
                if len(features_layer) == 0:
                    features_layer = feature
                    styles_layer = style
                else:
                    for i, (f, s) in enumerate(zip(feature, style)):
                        features_layer[i] = torch.cat([features_layer[i], f])
                        styles_layer[i] = torch.cat([styles_layer[i], s])
            
            targets.append(inputs['y'])
            domains.append(inputs['domain'])
        flat_list = [tensor for sublist in targets for tensor in sublist]
        targets = torch.cat(flat_list)
        # targets_name = [class_names[index] for index in targets]
        flat_list = [tensor for sublist in domains for tensor in sublist]
        domains = torch.cat(flat_list)
        # domains_name = [domain_names[index] for index in domains]

        side_length = 3  # 设置正方形的边长
        fig, axes = pyplot.subplots(2, len(features_layer), figsize=(len(features_layer) * side_length, 2 * side_length))

        for i, (f, s) in enumerate(zip(features_layer, styles_layer)):
            F = TSNE().fit(f.cpu())
            ax1 = fig.add_subplot(axes[0, i])
            scatter1 = ax1.scatter(F[:, 0], F[:, 1], 5, c=targets.cpu())
            ax1.set_xticks([])
            ax1.set_yticks([])
            
            S = TSNE().fit(s.cpu())
            ax2 = fig.add_subplot(axes[1, i])
            scatter2 = ax2.scatter(S[:, 0], S[:, 1], 5, c=domains.cpu())
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.set_xlabel(f'Stage {args.layer[i]}', fontsize=20)
            # 为散点图添加图例
            legend1 = ax1.legend(handles=scatter1.legend_elements()[0], title="Class", labels=class_names, loc=1)
            ax1.add_artist(legend1)  # 在同一个图中添加多个图例
            if i == len(features_layer) - 1:
                legend2 = ax2.legend(handles=scatter2.legend_elements()[0], title="Class", labels=class_names, loc=1)
            else:
                legend2 = ax2.legend(handles=scatter2.legend_elements()[0], title="Domain", labels=domain_names, loc=1)
            ax2.add_artist(legend2)  # 在同一个图中添加多个图例
        axes[0, 0].set_ylabel('Feature map', fontsize=20)
        axes[1, 0].set_ylabel('Style', fontsize=20)
        pyplot.tight_layout()
        pyplot.savefig(args.output_dir + "/mode" + str(args.mode) + 'TE' + str(test_envs) + "_plots.png", dpi=500)
        pyplot.clf()

def get_dataset(test_envs, dataset):
    in_splits = []
    out_splits = []
    for env_i, env in enumerate(dataset):
        # The split only depends on seed_hash (= trial_seed).
        # It means that the split is always identical only if use same trial_seed,
        # independent to run the code where, when, or how many times.
        out, in_ = split_dataset(
            env,
            int(0)
        )

        in_.transforms = {"x": DBT.basic}
        out.transforms = {"x": DBT.basic}
        in_splits.append((in_))
        out_splits.append((out))

    return dataset, in_splits, out_splits

if __name__ == "__main__":
    main()