import argparse
from glob import glob
from re import split
import torch
from domainbed.algorithms.algorithms import ERM, Contrast
from domainbed.lib import misc, utils
from domainbed.datasets import datasets
import matplotlib.pyplot as pyplot
from tqdm import tqdm
from domainbed.datasets import transforms as DBT
from torch.utils.data.dataloader import DataLoader
from openTSNE import TSNE

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def main():
    # network, output_dir
    parser = argparse.ArgumentParser(description="Domain generalization")
    parser.add_argument("--network", type=str, default="RN50")
    parser.add_argument("--output_dir", type=str, help="please add train_output pathdir to here, like 'train_output/OfficeHome/...'")
    parser.add_argument("--data_dir", type=str)

    args = parser.parse_args()

    timestamp = misc.timestamp()
    output = split('[. /]', args.output_dir)
    output = [item for item in filter(lambda x:x != '', output)]

    dataset = vars(datasets)[output[1]](args.data_dir)
    class_name = dataset.datasets[0].classes

    for pth in glob(args.output_dir + "/*.pth"):
        model = torch.load(pth)
        model.eval()
        sum = None
        targets = None
        name = split('[- . /]', pth)
        name = [item for item in filter(lambda x:x != '', name)]

        for env_i, env in enumerate(dataset):
            if ('te_' + dataset.environments[env_i]) != name[-3]:
                continue
            env.transform = DBT.basic
            test_loader = DataLoader(env, batch_size=32, shuffle=True, num_workers=4)
            with torch.no_grad():
                if isinstance(model, ERM):
                    net = model.featurizer
                elif isinstance(model, Contrast):
                    net = model.network

                for data, target in tqdm(test_loader):
                    data = data.to(device)
                    features = net(data)
                    if isinstance(model, Contrast):
                        features = features[1] # features = (image_text logits, image_features)
                    if sum != None:
                        sum = torch.cat((sum, features), 0)
                        targets = torch.cat((targets, target), 0)
                    else:
                        sum = features
                        targets = target
        X = TSNE().fit(sum.cpu())
        pyplot.scatter(X[:, 0], X[:, 1], 1, targets)
        pyplot.savefig(args.output_dir + "/" + name[-3] + ".png")
        # utils.plot(X, targets, save=args.output_dir + "/" + name[-3] + ".png")

if __name__ == "__main__":
    main()