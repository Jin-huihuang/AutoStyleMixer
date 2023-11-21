import argparse
from distutils.util import strtobool
from glob import glob
from re import split
import re
import sys
from matplotlib import pyplot as plt
import torch
from domainbed.algorithms import MSMT2
import numpy as np
from domainbed.datasets import datasets, split_dataset
from matplotlib.ticker import FormatStrFormatter  
import torch.nn.functional as F

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def main():
    # network, output_dir
    parser = argparse.ArgumentParser(description="Domain generalization")
    parser.add_argument("--network", type=str, default="RN50")
    parser.add_argument("--output_dir",
                        default='/export/home/zhh/project/MHDG/train_output/PACS/230823_10-29-18_PACS_MSMT2_100_3e-05R1_M/checkpoints',
                        type=str, help="please add train_output pathdir to here, like 'train_output/OfficeHome/...'")
    parser.add_argument("--data_dir", default='/root/data1/data', type=str)
    args = parser.parse_args()
    
    output = split('[. /]', args.output_dir)
    output = [item for item in filter(lambda x:x != '', output)]
    dataset = vars(datasets)[output[6]](args.data_dir)
    class_name = dataset.datasets[0].classes

    for pth in glob(args.output_dir + "/*.pth"):
        checkpoint = torch.load(pth)
        test_envs = checkpoint['test_envs']
        model = MSMT2(dataset.input_shape, dataset.num_classes, len(dataset) - len(test_envs), checkpoint['model_hparams']).to(device)
        model.load_state_dict(checkpoint['model_dict'], strict=False)
        # pic
        stylemixer = model.featurizer.network.stymix
        lmda1 = {}
        lmda2 = {}
        for name, module in stylemixer.items():
            lmda1[name] = F.softmax(module.lmda * 10000, dim=-1)[:,0].mean()
            lmda2[name] = F.softmax(module.lmda2 * 1000, dim=-1)[:,0].mean()
        fig, ax1 = plt.subplots()
        keys = list(lmda1.keys())
        keys = [re.search(r'\d+', key).group() for key in keys]
        values = [float(tensor.item()) for tensor in lmda1.values()]
        ax1.plot(keys, values, marker='o', linestyle='-', color='b', label='Probability')
        ax1.set_xlabel('Stage')
        ax1.set_ylabel('Probability')

        ax2 = ax1.twinx()
        keys = list(lmda2.keys())
        keys = [re.search(r'\d+', key).group() for key in keys]
        values = [float(tensor.item()) for tensor in lmda2.values()]
        ax2.plot(keys, values, marker='o', linestyle='-', color='r', label='Weight')
        ax2.set_ylabel('Weight')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2
        ax1.legend(lines, labels, loc='upper right')

        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # 显示图形
        plt.savefig(args.output_dir + "/" + 'TE' + str(test_envs) + ".png", dpi=500)
        plt.clf()

        
if __name__ == "__main__":
    main()