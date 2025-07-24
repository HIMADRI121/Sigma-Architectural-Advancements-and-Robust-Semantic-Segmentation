# Sigma-ArchitecturalAdvancements-and-Robust-Semantic-Segmentation

This repository contains the architectural changes for the paper Sigma: Siamese Mamba Network for Multi-Modal Semantic Segmentation.

<img width="1029" height="378" alt="image" src="https://github.com/user-attachments/assets/dcc2dd62-7411-41f6-9705-e7eed80bc464" />

Environment
We test our codebase with PyTorch 1.13.1 + CUDA 11.7 as well as PyTorch 2.2.1 + CUDA 12.1. Please install corresponding PyTorch and CUDA versions according to your computational resources. We showcase the environment creating process with PyTorch 1.13.1 as follows.

Create environment.

conda create -n sigma python=3.9
conda activate sigma
Install all dependencies. Install pytorch, cuda and cudnn, then install other dependencies via:

pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
Install Mamba

cd models/encoders/selective_scan && pip install . && cd ../../..


We use MFnet datasets, including both RGB-Thermal and RGB-Depth datasets.

Usage
Training
Please download the pretrained VMamba weights:

VMamba_Tiny.
VMamba_Small.
VMamba_Base.
Please put them under pretrained/vmamba/.

Config setting.

Edit config file in the configs folder.
Change C.backbone to sigma_tiny / sigma_small / sigma_base to use the three versions of Sigma.

Run multi-GPU distributed training:

NCCL_P2P_DISABLE=1 CUDA_VISIBLE_DEVICES="0,1,2,3" python -m torch.distributed.launch --nproc_per_node=4  --master_port 29502 train.py -p 29502 -d 0,1,2,3 -n "dataset_name"
Here, dataset_name=mfnet/pst/nyu/sun, referring to the four datasets.

You can also use single-GPU training:

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" torchrun -m --nproc_per_node=1 train.py -p 29501 -d 0 -n "dataset_name" 
Results will be saved in log_final folder.

Evaluation
Run the evaluation by:

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python eval.py -d="0" -n "dataset_name" -e="epoch_number" -p="visualize_savedir"
Here, dataset_name=mfnet/pst/nyu/sun, referring to the four datasets.
epoch_number refers to a number standing for the epoch number you want to evaluate with. You can also use a .pth checkpoint path directly for epoch_number to test for a specific weight.

If you want to use multi GPUs please specify multiple Device IDs:

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python eval.py -d="0,1,2,3,4,5,6,7" -n "dataset_name" -e="epoch_number" -p="visualize_savedir"
Results will be saved in log_final folder.

📈Results
Perfectly obtained the results on the original paper. And done various chnages in architecture for obtaining the hidden information and geeting some novality in the tasks we performed.
We provide our trained weights on the four datasets:

MFNet (9 categories)
Architecture	Backbone	mIOU	Weight
Sigma	VMamba-T	60.2%	Sigma-T-MFNet
Sigma	VMamba-S	61.1%	Sigma-S-MFNet
Sigma	VMamba-B	61.3%	Sigma-B-MFNet
