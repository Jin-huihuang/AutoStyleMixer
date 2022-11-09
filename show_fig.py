import argparse
import numpy as np
import torch
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Domain generalization")
parser.add_argument("--output_dir", type=str, help="please add train_output pathdir to here, like 'train_output/OfficeHome/...'")
args = parser.parse_args()

fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(2, 2, 1)
ax2 = fig.add_subplot(2, 2, 2)
ax3 = fig.add_subplot(2, 2, 3)
ax4 = fig.add_subplot(2, 2, 4)
# ax5 = fig.add_subplot(2, 3, 5)
# ax6 = fig.add_subplot(2, 3, 6)

# vector = torch.from_numpy(np.load("train_output/OfficeHome/221028_21-13-02_OH_Contrast_RN50_5e-05/text_vector.npy"))
# similarity = vector @ vector.t()
# sns.heatmap(similarity,ax=ax1,
#             annot=False, vmax=1, vmin=-1, xticklabels=False, yticklabels=False, square=True, cmap="coolwarm",linewidths=0.01)
# ax1.set_title('', fontsize=18)
# ax1.set_ylabel('', fontsize=18)
# ax1.set_xlabel('Text', fontsize=18)


vector = torch.from_numpy(np.load("train_output/OfficeHome/221026_16-11-41_OH_ERM_RN50_1e-06/te_Clipart_image_vector.npy"))
vector = vector / vector.norm(dim=1, keepdim=True)
similarity = vector @ vector.t()
sns.heatmap(similarity,ax=ax1,
            annot=False, vmax=1, vmin=-1, xticklabels=False, yticklabels=False, square=True, cmap="coolwarm",linewidths=0.01)
ax1.set_title('', fontsize=18)
ax1.set_ylabel('', fontsize=18)
ax1.set_xlabel('CLIP-pre + ERM', fontsize=20)


vector = torch.from_numpy(np.load('train_output/OfficeHome/221107_12-56-58_OH_Contrast_RN50_1e-06/te_Clipart_image_vector.npy'))
vector = vector / vector.norm(dim=1, keepdim=True)
similarity = vector @ vector.t()
sns.heatmap(similarity,ax=ax2,
            annot=False, vmax=1, vmin=-1, xticklabels=False, yticklabels=False, square=True, cmap="coolwarm",linewidths=0.01)
ax2.set_title('', fontsize=18)
ax2.set_ylabel('', fontsize=18)
ax2.set_xlabel('CLIP-pre + Text', fontsize=20)


vector = torch.from_numpy(np.load('train_output/OfficeHome/221106_20-14-02_OH_Contrast_RN50_5e-05/te_Clipart_image_vector.npy'))
# vector = vector / vector.norm(dim=1, keepdim=True)
similarity = vector @ vector.t()
sns.heatmap(similarity,ax=ax4,
            annot=False, vmax=1, vmin=-1, xticklabels=False, yticklabels=False, square=True, cmap="coolwarm",linewidths=0.01)
ax4.set_title('', fontsize=18)
ax4.set_ylabel('', fontsize=18)
ax4.set_xlabel('ImageNet-pre + Text', fontsize=20)


vector = torch.from_numpy(np.load('train_output/OfficeHome/221030_15-40-50_OH_ERM_RN50_5e-05/te_Clipart_image_vector.npy'))
vector = vector / vector.norm(dim=1, keepdim=True)
similarity = vector @ vector.t()
sns.heatmap(similarity,ax=ax3,
            annot=False, vmax=1, vmin=-1, xticklabels=False, yticklabels=False, square=True, cmap="coolwarm",linewidths=0.01)
ax3.set_title('', fontsize=18)
ax3.set_ylabel('', fontsize=18)
ax3.set_xlabel('ImageNet + ERM', fontsize=20)


# vector = torch.from_numpy(np.load('train_output/OfficeHome/221028_21-13-02_OH_Contrast_RN50_5e-05/te_Clipart_image_vector.npy'))
# # vector = vector / vector.norm(dim=1, keepdim=True)
# similarity = vector @ vector.t()
# sns.heatmap(similarity,ax=ax6,
#             annot=False, vmax=1, vmin=-1, xticklabels=False, yticklabels=False, square=True, cmap="coolwarm",linewidths=0.01)
# ax6.set_title('', fontsize=18)
# ax6.set_ylabel('', fontsize=18)
# ax6.set_xlabel('ImageNet + Text + ERM (ours)', fontsize=20)

plt.savefig(args.output_dir + "/" "4.png")

