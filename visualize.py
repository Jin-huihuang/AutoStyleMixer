import argparse
from distutils.util import strtobool
from glob import glob
from re import split
from time import pthread_getcpuclockid
import torch
from domainbed.algorithms.algorithms import ERM, Contrast
from domainbed.networks import CLIP
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
    parser.add_argument("--domain", type=int, default=0, help='0:source_only, 1:target_only, 2:all')
    parser.add_argument("--CLIP", type=strtobool, default=False)
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
            if args.domain == 0: # source only
                if ('te_' + dataset.environments[env_i]) == name[-3]:
                    continue
            elif args.domain == 1: # target only 
                if ('te_' + dataset.environments[env_i]) != name[-3]:
                    continue
            elif args.domain == 2: # all
                pass
            else:
                print("domain should be set 0 or 1 or 2")
            env.transform = DBT.basic
            test_loader = DataLoader(env, batch_size=32, shuffle=False, num_workers=4)
            with torch.no_grad():
                if isinstance(model, ERM):
                    net = model.featurizer
                elif isinstance(model, Contrast):
                    net = model.network

                for data, target in tqdm(test_loader):
                    data = data.to(device)
                    if isinstance(net, CLIP):
                        features = net.network.encode_image(data)
                    else:
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
        class_sum = torch.zeros(targets.unique().size())
        center_avg = torch.zeros([class_sum.size(0),2])
        for i in range(X.shape[0]):
            class_sum[targets[i]] += 1
            center_avg[targets[i]] += X[i]
        phi = torch.zeros(class_sum.size(0))
        for i in range(class_sum.size(0)):
            center_avg[i] /= class_sum[i]
        for i in range(X.shape[0]):
            phi[targets[i]] += torch.norm(torch.from_numpy(X[i] - center_avg[targets[i]].numpy()))
        phi = phi / class_sum
        print(str(phi.mean()))
        pyplot.scatter(X[:, 0], X[:, 1], 1, targets)
        pyplot.savefig(args.output_dir + "/" + str(args.domain)+ name[-3] + str(phi.mean()) + ".png")
        # utils.plot(X, targets, save=args.output_dir + "/" + name[-3] + ".png")

if __name__ == "__main__":
    main()