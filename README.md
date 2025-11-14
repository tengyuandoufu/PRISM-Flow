# PRISM-Flow：Unsupervised Cross-Modal Medical Image Registration via Optical Flow with Geometric Prior Guidance
There is no time to release the code now because there are still supplementary materials that have not been written. We will wait for this month and the code will be released gradually.
To facilitate broader reproducibility among Windows users, our code has been designed to operate natively on the Windows platform, eliminating the need for a Linux environment.
# 🔨 1.Setup
## 1.1 Requirements
To set up the environment. It is not necessary to run the application on Linux. We provide support for running it on Windows, and the system does not require an extensive amount of video memory—approximately 15 GB is sufficient.
```bash
conda create -n PRISM-Flow python=3.10
conda activate PRISM-Flow
pip install -r requirements.txt
```
# 📁 2.Dataset Preparation
This project uses data from the [ADNI-4 dataset](https://adni.loni.usc.edu/)  and [SynthRAD2025 Grand Challenge dataset](https://zenodo.org/records/14918089).
## 2.1 SynthRAD2025 dataset
- Task 1 (MRI-to-CT) 
- Task 2 (CBCT-to-CT)
- Within each task, cases are categorized into three anatomical regions:Head-and-neck (HN),Thorax (TH),and Abdomen (AB)
## 2.2 ADNI-4 dataset
# 🔄 3.Preprocessing

# 🚂 4.Training
```bash
Liunx: sh scripts/train.sh
Windows: python train.py --gpu_ids=0(If you have more GPU, you can 0,1,2···)
```
# ⚗️ 5.Testing
```bash
Liunx: sh scripts/test.sh
Windows: python test.py --gpu_ids=0
