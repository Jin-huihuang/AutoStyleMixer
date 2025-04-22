# Auto-StyleMixer

👨‍🔬 By Huihuang Zhang, Haigen Hu*, Xiaoqin Zhang, Bin Cao

> Existing domain generalization (DG) approaches that rely on traditional techniques like the Fourier transform and normalization can extract style information for cross-domain data augmentation by confusing styles to enhance model generalization. However, these one-to-one methods face two significant challenges: 1) They cannot effectively extract pure style information in deep layers, potentially disrupting the ability to learn content information. 2) Due to the unknown purity of the extracted style information, considerable resources are required to find the optimal style-mixing configuration based on manual experience. To address these challenges, we propose a universal N-to-one cross-domain data augmentation framework, named Auto-StyleMixer, which not only extracts purer style information but also adapts to learn style-mixing configurations without any manual intervention. The proposed framework can embed any traditional style extraction techniques and can be integrated as a plug-and-play module into any architecture, whether CNNs or Transformers. Extensive experiments demonstrate the effectiveness of the proposed method, showing that it achieves state-of-the-art performance on five DG benchmarks.

<p align="center">
  <img src="assets/image.png" width="80%" />
</p>

This repository contains the official implementation for the KBS paper
[*Auto-StyleMixer: A Generalized Framework for Cross-domain Data Augmentation*]()

---

## ⚙️ Environment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Download datasets:

```bash
python -m domainbed.scripts.download --data_dir=/data/
```

Environment details:

```
Python:     3.7.13  
PyTorch:    1.12.0+cu113  
Torchvision:0.13.0+cu113  
CUDA:       11.3  
CUDNN:      8.2  
NumPy:      1.21.5  
PIL:        9.0.1
```

---

## 🚀 Run Training

You can train Auto-StyleMixer using the `train_all.py` script for multiple leave-one-domain-out cross-validations:

```bash
python train_all.py PACS --dataset PACS --data_dir /data/
```

### 📌 Example Usage per Dataset

#### PACS

```bash
CUDA_VISIBLE_DEVICES=0 python train_all.py PACS --dataset PACS --deterministic \
--trial_seed 0 --checkpoint_freq 200 --steps 5000 --data_dir $data_dir --algorithm AutoSM \
--lr 3e-5 --backbone ResNet50 --MT True --AdaptiveAug True --allKL True --method N \
--mix_layers "['conv1','conv2_x','conv3_x','conv4_x','conv5_x']" --Mix_T 100
```

#### VLCS

```bash
CUDA_VISIBLE_DEVICES=0 python train_all.py VLCS --dataset VLCS --deterministic \
--trial_seed 0 --checkpoint_freq 200 --steps 5000 --data_dir $data_dir --algorithm AutoSM \
--lr 1e-6 --backbone ResNet50 --MT True --AdaptiveAug True --allKL True --method N \
--mix_layers "['conv1','conv2_x','conv3_x','conv4_x','conv5_x']" --Mix_T 100
```

#### OfficeHome

```bash
CUDA_VISIBLE_DEVICES=0 python train_all.py OH --dataset OfficeHome --deterministic \
--trial_seed 0 --checkpoint_freq 200 --steps 5000 --data_dir $data_dir --algorithm AutoSM \
--lr 3e-5 --backbone ResNet50 --MT True --AdaptiveAug True --allKL True --method N \
--mix_layers "['conv1','conv2_x','conv3_x','conv4_x','conv5_x']" --Mix_T 100
```

#### TerraIncognita

```bash
CUDA_VISIBLE_DEVICES=0 python train_all.py TR0 --dataset TerraIncognita --deterministic \
--trial_seed 0 --checkpoint_freq 200 --steps 5000 --data_dir $data_dir --algorithm AutoSM \
--lr 3e-5 --backbone ResNet50 --MT True --AdaptiveAug True --allKL True --method N \
--mix_layers "['conv1','conv2_x','conv3_x','conv4_x','conv5_x']" --Mix_T 100
```

#### DomainNet

```bash
CUDA_VISIBLE_DEVICES=0 python train_all.py DomainNet --dataset DomainNet --deterministic \
--trial_seed 0 --checkpoint_freq 1000 --steps 15000 --data_dir $data_dir --algorithm AutoSM \
--lr 3e-5 --backbone ResNet50 --MT True --AdaptiveAug True --allKL True --method N \
--mix_layers "['conv1','conv2_x','conv3_x','conv4_x','conv5_x']" --Mix_T 1000
```

| Argument        | Description                                                                  |
| --------------- | ---------------------------------------------------------------------------- |
| `--MT`          | Enable Mean Teacher for temporal ensembling during training.                 |
| `--AdaptiveAug` | Enable AdaptiveAug for learning layer-wise adaptive style mixing.            |
| `--AdaptiveP`   | Enable adaptive probability adjustment for augmentation.                     |
| `--AdaptiveW`   | Enable adaptive weight adjustment for style mixing.                          |
| `--allKL`       | Enable Multiple Contrast Learning (MCL) loss for reducing domain gap.        |
| `--method`      | Cross-domain data augmentation method: `N` (Normalization) or `F` (Fourier). |
| `--mix_layers`  | List of model layers to apply StyleMixer, e.g., `['conv1','conv2_x',...]`.   |
| `--Mix_T`       | Temperature parameter for AdaptiveAug controlling the softmax sharpness.     |

## 📚 Citation

If you find this repository helpful, please consider citing:

```bibtex

```
