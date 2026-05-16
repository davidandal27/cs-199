# ASVspoof5 Experiment


Welcome to the official release of pretrained models and scripts for *'Nes2Net: a lightweight nested architecture designed for foundation model-driven speech anti-spoofing'* [![arXiv](https://img.shields.io/badge/arXiv-2504.05657-b31b1b.svg)](https://arxiv.org/abs/2504.05657)

Accpeted to: **IEEE Transactions on Information Forensics and Security** (T-IFS), IEEE Link: https://ieeexplore.ieee.org/document/11222612

## 📁 Supported Datasets
- **asvspoof5 Branch (Current Branch)**: For the **ASVspoof 5** dataset  
  
- **main Branch**: For **ASVspoof 2019/2021** and **In-the-Wild** datasets: 👉 [main branch](https://github.com/Liu-Tianchi/Nes2Net_ASVspoof_ITW/tree/main)

- **Controlled Singing Voice Deepfake Detection (CtrSVDD)**: 👉 [View here](https://github.com/Liu-Tianchi/Nes2Net)
  
## Pretrained Model
We have uploaded pretrained models of our experiments. You can download pretrained models from [OneDrive](https://entuedu-my.sharepoint.com/:u:/g/personal/truongdu001_e_ntu_edu_sg/EaBasBsecRpErWEVzXdht7cBiWWYFuLTeXt11ABbHX9yBg?e=vxVOOI). 

## Setting up environment
```
conda create --name asvspoof5 python=3.9
conda activate asvspoof5
conda install pytorch==1.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Usage
Before running the experiment, replace the data directory of `database_path` in the config file you plan to use, such as `./config/WavLM_Nes2Net_ASVspoof5.conf` or `./config/WavLM_AASIST_ASVspoof5.conf`.

### Single-GPU / local launch
Use the plain Python entrypoint for local or single-GPU runs:

```bash
python -m training.main \
  --config ./config/WavLM_Nes2Net_ASVspoof5.conf \
  --defense-config ./config/defense.conf
```

### Multi-GPU HPC launch (`torchrun`)
For the single-node `4`-GPU DDP path, launch with `torchrun` and treat `--batch-size` as a per-process value. For the first pass, use `8` per GPU:

```bash
torchrun --standalone --nproc_per_node=4 -m training.main \
  --config ./config/WavLM_Nes2Net_ASVspoof5.conf \
  --defense-config ./config/defense.conf \
  --batch-size 8
```

See [docs/TRAINING_MAIN_HPC.md](docs/TRAINING_MAIN_HPC.md) for the full HPC procedure, path overrides, verification checklist, and startup failure conditions.

This automatically picks the last .pth file generated
```
CHECKPOINT=$(ls -t ./outputs/WavLM_Nes2Net/checkpoints/*.pth | head -1)
```

To run clean eval-only scoring from a pretrained checkpoint:
```bash
python -m attacks.eval_fgsm \
  --config ./config/WavLM_Nes2Net_ASVspoof5.conf \
  --defense-config ./config/defense.conf \
  --weights "$CHECKPOINT" \
  --output-dir /path/to/output \
  --dataset-root /path/to/data \
  --audio-root /path/to/eval_audio \
  --trial-file /path/to/ASVspoof5.eval.track_1.tsv \
  --ssl-pretrained-path /path/to/WavLM-Large.pt \
  --split eval
```

This eval-only path is intended for Colab and Google Drive backed assets, and it writes a score file plus a metric report without entering the training loop.

To run matched clean and FGSM scoring from the same ordered trial list:
```bash
python -m attacks.eval_fgsm \
  --config ./config/WavLM_Nes2Net_ASVspoof5.conf \
  --defense-config ./config/defense.conf \
  --weights "$CHECKPOINT" \
  --output-dir /path/to/output \
  --dataset-root /path/to/data \
  --audio-root /path/to/eval_audio \
  --trial-file /path/to/ASVspoof5.eval.track_1.tsv \
  --ssl-pretrained-path /path/to/WavLM-Large.pt \
  --split eval \
  --epsilon 0.001 \
  --batch-size 8 \
  --save-adv-audio
```

If your audio and TSV files are not under the same dataset root, pass `--audio-root` to the directory that actually contains the `.flac` files for the active split and `--trial-file` to the exact TSV you want to score. `--dataset-root` can still point to any existing dataset base directory for compatibility with the config.

To run matched clean and PGD scoring from the same ordered trial list:
```bash
python -m attacks.eval_pgd \
  --config ./config/WavLM_Nes2Net_ASVspoof5.conf \
  --weights /path/to/checkpoint.pth \
  --output-dir /path/to/output \
  --dataset-root /path/to/data \
  --audio-root /path/to/eval_audio \
  --trial-file /path/to/ASVspoof5.eval.track_1.tsv \
  --ssl-pretrained-path /path/to/WavLM-Large.pt \
  --split eval \
  --epsilon 0.001 \
  --steps 5 \
  --batch-size 8
```

PGD in this repo is an untargeted `Linf` waveform attack that iterates projected updates over multiple steps. Compared with FGSM, it is slower but typically a stronger attack because it reuses gradients several times instead of taking a single step.

Practical default recommendation:

- start with `PGD-5` using `--epsilon 0.001 --steps 5`
- leave `--alpha` unset unless you need manual control, because the CLI derives it as `epsilon / steps`
- keep `--random-start` off for the first pass so repeated runs stay easier to compare
- only increase `--steps` when there is a concrete reason to pay for longer runtimes

Runtime guidance for large evaluation sets:

- start with `--split dev` as a dry run before launching the full eval split
- reduce `--batch-size` first if you hit GPU memory limits
- keep `--save-adv-audio` off unless you explicitly need waveform inspection, because it increases disk I/O and runtime
- use `--metrics-only` when you only need the computed comparisons and do not want score, metric, or summary files written

The PGD run writes artifacts under `output/<model_tag>/<split>_pgd_eval/`. By default this includes:

- `clean_scores.txt`
- `pgd_eps_<epsilon>_steps_<steps>_scores.txt`
- `clean_metrics.txt`
- `pgd_eps_<epsilon>_steps_<steps>_metrics.txt`
- `pgd_metrics_summary.json`
- `pgd_metrics_summary.txt`

## Colab FGSM Workflow

The repository includes [`notebooks/fgsm_eval_colab.ipynb`](notebooks/fgsm_eval_colab.ipynb) for a step-by-step Google Colab flow. The notebook is organized as:

- mount Google Drive
- enter the repo and install dependencies
- define Drive-backed dataset, checkpoint, backbone, and output paths
- validate those paths before scoring
- run clean evaluation
- run FGSM scoring
- inspect the clean vs adversarial summary artifacts

Minimal Colab CLI example with explicit Drive-backed paths:
```bash
python -m attacks.eval_fgsm \
  --config ./config/WavLM_Nes2Net_ASVspoof5.conf \
  --defense-config ./config/defense.conf \
  --weights "/content/drive/MyDrive/Education/Subjects/CS 199: Special Problems II/project_storage/checkpoints/model.pth" \
  --output-dir "/content/drive/MyDrive/Education/Subjects/CS 199: Special Problems II/project_storage/outputs" \
  --dataset-root "/content/drive/MyDrive/Education/Subjects/CS 199: Special Problems II/project_storage/data" \
  --metadata-root "/content/drive/MyDrive/Education/Subjects/CS 199: Special Problems II/project_storage/data" \
  --ssl-pretrained-path "/content/drive/MyDrive/Education/Subjects/CS 199: Special Problems II/project_storage/pretrained_models/WavLM-Large.pt" \
  --split eval \
  --epsilon 0.001
```

## Colab PGD Workflow

The repository includes [`notebooks/pgd_eval_colab.ipynb`](notebooks/pgd_eval_colab.ipynb) for the matching Google Colab PGD flow. The notebook is organized as:

- mount Google Drive
- enter the repo and install dependencies
- define Drive-backed dataset, checkpoint, backbone, and output paths
- validate those paths before scoring
- run clean evaluation
- run PGD scoring with the practical `PGD-5` baseline
- inspect the clean vs adversarial JSON and text summaries

Minimal Colab CLI example with explicit Drive-backed paths:
```bash
python -m attacks.eval_pgd \
  --config ./config/WavLM_Nes2Net_ASVspoof5.conf \
  --weights "/content/drive/MyDrive/Education/Subjects/CS 199: Special Problems II/project_storage/checkpoints/model.pth" \
  --output-dir "/content/drive/MyDrive/Education/Subjects/CS 199: Special Problems II/project_storage/outputs" \
  --dataset-root "/content/drive/MyDrive/Education/Subjects/CS 199: Special Problems II/project_storage/data" \
  --metadata-root "/content/drive/MyDrive/Education/Subjects/CS 199: Special Problems II/project_storage/data" \
  --ssl-pretrained-path "/content/drive/MyDrive/Education/Subjects/CS 199: Special Problems II/project_storage/pretrained_models/WavLM-Large.pt" \
  --split eval \
  --epsilon 0.001 \
  --steps 5
```

For a dry-run checklist and expected PGD artifacts, see [`docs/PGD_VERIFICATION.md`](docs/PGD_VERIFICATION.md).

## Repository Layout
- `training/`: training and evaluation entrypoints
- `attacks/`: adversarial attack and attack-evaluation entrypoints
- `src/`: helper modules for datasets, augmentation, path resolution, and training utilities
- `models/`: model architecture definitions
- `eval/`: evaluation metric helpers
- `config/`: experiment configuration files
- `docs/`: reference documentation and paper artifacts
- `notebooks/`: Colab and notebook workflows

### Acknowledge
Our work is built upon the [Baseline-AASIST](https://github.com/asvspoof-challenge/asvspoof5/tree/main/Baseline-AASIST) We also follow some parts of the following codebases:

[HM-Conformer](https://github.com/talkingnow/HM-Conformer/tree/main) (for noise augmentation).

[unilm](https://github.com/microsoft/unilm) (for WavLM model).

## Citation
```
@ARTICLE{Nes2Net,
  author={Liu, Tianchi and Truong, Duc-Tuan and Das, Rohan Kumar and Lee, Kong Aik and Li, Haizhou},
  journal={IEEE Transactions on Information Forensics and Security}, 
  title={Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-Spoofing}, 
  year={2025},
  volume={20},
  number={},
  pages={12005-12018},
  keywords={Foundation models;Feature extraction;Computational modeling;Computer architecture;Computational efficiency;Dimensionality reduction;Acoustics;Kernel;Robustness;Deepfakes;Deepfake detection;speech anti-spoofing;Res2Net;Nes2Net;SSL;speech foundation model},
  doi={10.1109/TIFS.2025.3626963}}

```
