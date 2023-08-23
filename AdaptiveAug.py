import argparse
from distutils.util import strtobool
from glob import glob
from re import split
import sys
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
    args = parser.parse_args()
    
    output = split('[. /]', args.output_dir)
    output = [item for item in filter(lambda x:x != '', output)]
    dataset = vars(datasets)[output[6]](args.data_dir)
    class_name = dataset.datasets[0].classes

    for pth in glob(args.output_dir + "/*.pth"):
        checkpoint = torch.load(pth)
        test_envs = checkpoint['test_envs']
        model = MSMT2(dataset.input_shape, dataset.num_classes, len(dataset) - len(test_envs), checkpoint['model_hparams']).to(device)
        model.load_state_dict(checkpoint['model_dict'])
        # pic
        lmda1 = model
        lmda2 = model
        
if __name__ == "__main__":
    main()