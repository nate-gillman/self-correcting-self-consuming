## Self-Correcting Self-Consuming Loops for Generative Model Training


[![arXiv](https://img.shields.io/badge/arXiv-2402.07087-<COLOR>.svg)](https://arxiv.org/abs/2402.07087)


The official PyTorch implementation of the paper [**"Self-Correcting Self-Consuming Loops for Generative Model Training"**](https://arxiv.org/abs/2402.07087), which has been accepted at [**ICML 2024**](https://icml.cc/Conferences/2024).
Please visit our [**webpage**](https://nategillman.com/sc-sc.html) for more details.


![teaser](assets/motion_null.gif)




## Recreating results from paper

<details>
  <summary><b> Environment setup </b></summary>

<br>

The main building blocks for this repo include [Human Motion Diffusion Model](https://guytevet.github.io/mdm-page/), [Universal Humanoid Controller](https://github.com/ZhengyiLuo/UHC), [VPoser](https://github.com/nghorbani/human_body_prior).
Please visit their webpages for more details, including license info.
Note that their code depends on other libraries, including CLIP, SMPL, SMPL-X, PyTorch3D, and uses datasets that each have their own respective licenses that must also be followed.

### Step 1: build conda env

Run the script: `./setup.sh`.

This will create a conda virtual environment and perform a basic test (`test_environment.py`) to see if all succeeds. 

The environment setup has several major steps which depend greatly on the host machine. While `setup.sh` aspires to be robust / 'just work', there will be differences from system to system. 
For completeness, those steps are:
1. Create a Python 3.8.12 conda virtual environment named `"scsc"`
2. Install the dependencies of [MDM](https://github.com/GuyTevet/motion-diffusion-model)
3. Install the dependencies of [UHC](https://github.com/ZhengyiLuo/UHC) (including [Mujoco](https://github.com/openai/mujoco-py), which requires [Boost](https://www.boost.org/) to cythonize.)
4. Install visualization dependencies ([Body Visualizer](https://github.com/nghorbani/body_visualizer), [VPoser](https://github.com/nghorbani/human_body_prior))

Optionally, the last step of `./setup.sh` will facilitate moving the SMPL, SMPL+H, and SMPL+X models into their expected locations.

You must have an account on the following websites **AND AGREE TO THEIR TERMS AND CONDITIONS**:
- SMPL-X: https://smpl-x.is.tue.mpg.de/register.php
- SMPL: https://smpl.is.tue.mpg.de/modellicense.html
- SMPL-H: https://mano.is.tue.mpg.de/register.php

The data download script may also be run independently of the model setup. One can run: `./get_smpl_data.sh`.
More detail on the data dependencies can be found in Step 3.

<!-- <details>
  <summary><b> (Optional) basic environment setup test, to check if sampling works </b></summary> 

Running a sampling and basic visualization is a good intermediate test to run at this point. To do this,
first download the [`humanml-encoder-512`](https://github.com/GuyTevet/motion-diffusion-model?tab=readme-ov-file#3-download-the-pretrained-models) model from the MDM repository. 

Place it into `save/`. Make a copy of the model weights you wish to use for MDM (e.g., `model000200000.pt`) and
name it `model.pt`. The save directory should now look like this:
```commandline
|── save
└── humanml_trans_enc_512
    ├── args.json
    ├── eval_humanml_trans_enc_512_000475000_gscale2.5_wo_mm.log
    ├── model000200000.pt
    ├── model000475000.pt
    ├── model.pt
    ├── opt000200000.pt
    └── opt000475000.pt
```

To run the visualization test, enter the following into your shell:
```bash
conda activate scsc
mkdir -p test_videos
python visualize/visualizer.py
```

Examine the contents of the `test_videos` directory to find the results. 

This will output should look something like this:
```commandline
(scsc) [user@host]$ python visualize/visualizer.py WARNING: You are using a SMPL model, with only 10 shape coefficients.
usage: visualizer.py [-h] model_path
visualizer.py: error: the following arguments are required: model_path
WARNING: !-model_path was not specified through args.
Reading ././dataset/humanml_opt.txt
Loading dataset t2m !!...
100%|████████████████████████████████| 1460/1460 [00:02<00:00, 621.19it/s] num_synthetic_examples_included = 0
Skipped: 0 samples. Total Samples: 1404 Number of diffusion steps: 1000 Sampling batch 1/1!!...
Sampler batch size: 64
Sampling [repetitions: 0]
100%|████████████████████████████████| 1000/1000 [01:44<00:00, 9.54it/s]
Processing Sample [repetitions: 0] data rep: hml_vec.....
hml vec rep
Successfully saved wireframe videos to:
[PosixPath('test_videos/wireframe_video_prompt_0.mp4'), PosixPath('test_videos/wireframe_video_prompt_1.mp4')]
POSE_SEQ shape (296, 22, 3)
2024-02-26 00:53:21.608 | INFO | translation.utils_mdm_to_amass.human_body_prior.src.human_body_prior.tools.model_loader:load_model:97 - Loaded model in eval mode with trained weights: ./body_models/vposer_v2_05/snapshots/V02_05_epoch=13_val_loss=0.03.ckpt
VPoser Advanced IK: 100%|████████████████████████████| 1/1 [00:13<00:00, 13.70s/it] 
100%|██████████████████████████████| 196/196 [01:10<00:00, 2.76it/s] 
100%|██████████████████████████████| 100/100 [00:35<00:00, 2.78it/s] 
Successfully saved skinned videos to:
[PosixPath('test_videos/skinned_video_0.gif'), PosixPath('test_videos/skinned_video_1.gif')]
```

</details> -->

### Step 2: obtain HumanML3D dataset, and filter it to obtain our subset

First, must build [HumanML3D](https://github.com/EricGuo5513/HumanML3D) dataset.

Instructions on how to build the dataset may be found here: [LINK](https://github.com/EricGuo5513/HumanML3D?tab=readme-ov-file#how-to-obtain-the-data)

To obtain the AMASS data required to build HumanML3D, one can use the `get_smpl_data.sh` script and then the `extract_humanml_3d.sh` script. Together, this will download the required AMASS datasets and put them in a convenient location to proceed with HumanML3D's instructions. 
Again, usage of this script requires one have an active account on the [AMASS website](https://amass.is.tue.mpg.de/index.html) and agree to the license of all individual datasets. 

```bash
git clone https://github.com/EricGuo5513/HumanML3D.git
# follow HumanML3D setup instructions at the above repo, then
cp -r HumanML3D/HumanML3D ./dataset/HumanML3D
cp HumanML3D/index.csv ./dataset/HumanML3D/index.csv
```

Then, at [BMLMoVi](https://borealisdata.ca/dataset.xhtml?persistentId=doi%3A10.5683%2FSP2%2FJRHDRN&version=&q=&fileTypeGroupFacet=%22Archive%22&fileAccess=&fileTag=&fileSortField=&fileSortOrder=), you need to download and unpack the files:
- `F_Subjects_1_45.tar`: [LINK](https://borealisdata.ca/file.xhtml?fileId=128299&version=5.0)
- `F_Subjects_46_90.tar`: [LINK](https://borealisdata.ca/file.xhtml?fileId=92072&version=5.0)

and put their contents together inside the folder `dataset/F_Subjects_1_90` at the root of the repository.
We use this when we run the following script, to filter the HumanML3D dataset into smaller subdata sets of sizes $\{64, 128, 256, 2794\}$ as described in the paper.

```bash
python exp_scripts/filter_dataset.py
```

### Step 3: Download dependencies for MDM, UHC, and inverse kinematics engine


```bash
# from original MDM repo
pip install gdown
bash prepare/download_glove.sh
bash prepare/download_smpl_files.sh
bash prepare/download_t2m_evaluators.sh
```

The `download_smpl_files.sh` will place files inside `body_models/smpl`.
Then, download and place these files in the repo as indicated:
- [DMPL model](https://smpl.is.tue.mpg.de/download.php) (go to downloads, then "Download DMPLs compatible with SMPL", then put `dmpls` folder inside `body_models` directory)
- [VPoser v2.0](https://smpl-x.is.tue.mpg.de/) (sign up for an account and find the VPoser v2 download in the 'Downloads tab') and unzip, then place it in `body_models/vposer_v2_05` (i.e. rename downloaded folder to `vposer_v2_05`) 
- [SMPL-H model](https://mano.is.tue.mpg.de/) (find the Extended SMPL+H model download in the 'Downloads tab') and place the `smplh` folder in `body_models`


After all this, `body_models` directory should look like this:

  ```bash
body_models
├── dmpls
│   ├── female
│       ├── model.npz
│   ├── male
│       ├── model.npz
│   ├── neutral
│       ├── model.npz
├── smpl
│   ├── J_regressor_extra.npy
│   ├── kintree_table.pkl
│   ├── SMPL_NEUTRAL.pkl
│   ├── smplfaces.npy
├── smplh
│   ├── female
│       ├── model.npz
│   ├── male
│       ├── model.npz
│   ├── neutral
│       ├── model.npz
├── vposer_v2_05
│   ├── snapshots
│       ├── V02_05_epoch=08_val_loss=0.03.ckpt
│       ├── V02_05_epoch=13_val_loss=0.03.ckpt
│   ├── V02_05.yaml
```


Also need data for UHC:

```bash
# Also need data for UHC
cd UniversalHumanoidControl
bash download_data.sh
```

</details>


<details>
  <summary><b> Running the self-consuming loop experiments: Gaussian toy example </b></summary>

<br>

```bash
python exp_scripts/gaussian_toy_example.py
```

</details>

<details>
  <summary><b> Running the self-consuming loop experiments: MNIST toy example </b></summary>

<br>

Create a separate conda env for these experiments:

```bash
conda create -n mnist_toy python=3.11
conda activate mnist_toy

conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install tqdm matplotlib -y
conda install scikit-learn -y
```

Train an image classifer for MNIST digits; the learned embeddings will be used to compute the FID scores later.

```bash
mkdir -p exp_outputs/mnist
python exp_scripts/mnist/fid_lenet.py
```

This first training script trains the baseline, which is generations 0 through 50.
The last checkpoint from Generation 0 will  be used to seed all of the self-consuming experiments.
Don't start the other runs until this run finishes.

```bash
NUM_EPOCH=20

python exp_scripts/mnist/self_consuming_ddpm_mini.py \
    --n_epoch_for_training_from_scratch ${NUM_EPOCH} \
    --train_type baseline \
    --synth_aug_percent 0.0 \
    --fraction_of_train_set_to_train_on 0.2 \
    --save_dir_parent ./exp_outputs/mnist/ \
    --lr_divisor 20 \
    --resume_starting_at_generation 0
```

This script trains the self-consuming loop.
To recreate the results from the paper, you should run this script four times, for each `SYNTH_AUG_PERCENT` in `{0.2, 0.5, 1.0, 1.5}`.
These can all be run in parallel.

```bash
NUM_EPOCH=20
SYNTH_AUG_PERCENT=0.2
python exp_scripts/mnist/self_consuming_ddpm_mini.py \
    --n_epoch_for_training_from_scratch ${NUM_EPOCH} \
    --train_type iterative_finetuning \
    --synth_aug_percent ${SYNTH_AUG_PERCENT} \
    --fraction_of_train_set_to_train_on 0.2 \
    --save_dir_parent ./exp_outputs/mnist/ \
    --lr_divisor 20 \
    --resume_starting_at_generation 0
```

And this script trains the self-consuming loop with self-correction.
Again, to recreate the results from the paper, you should run this script four times, for each `SYNTH_AUG_PERCENT` in `{0.2, 0.5, 1.0, 1.5}`.
These can also all be run in parallel.

```bash
NUM_EPOCH=20
SYNTH_AUG_PERCENT=0.2
python exp_scripts/mnist/self_consuming_ddpm_mini.py \
    --n_epoch_for_training_from_scratch ${NUM_EPOCH} \
    --train_type iterative_finetuning_with_correction \
    --synth_aug_percent ${SYNTH_AUG_PERCENT} \
    --fraction_of_train_set_to_train_on 0.2 \
    --n_clusters_per_digit 16 \
    --save_dir_parent ./exp_outputs/mnist/ \
    --lr_divisor 20 \
    --resume_starting_at_generation 0
```

At any point during training, you can check on progress by running the below script.
It will generate graphs and write them to `exp_outputs/mnist/graphs`.

```bash
python exp_scripts/mnist/generate_graphs.py ./exp_outputs/mnist
```


</details>


<details>
  <summary><b> Running the self-consuming loop experiments: human motion generation</b></summary>

<br>

The bash scripts below can be run without any changes.
If your compute resources are managed by Slurm, then you might consider taking a look at 
the Slurm script that we used, which is provided at `exp_scripts/slurm.sh`.
You would need to change the resource requests and environment to match whatever your slurm setup is,
and of course you would need to change the last line, which executes the bash script listed below.


<details style="margin-left: 20px;"><summary><b>dataset size = 64</b></summary>


#### $n = 64$, training from scratch


```bash
# STEP 1: we train generation 0 on just ground truth data
bash exp_scripts/dataset_0064/train_generation_0.sh

# STEP 2: copy the checkpoint from that experiment to seed all the other experiments
python exp_scripts/dataset_0064/copy_generation_0.py

# STEP 3: After the above scripts finish, each of following 9 scripts can run in parallel

# STEP 3A: we train the baseline model
bash exp_scripts/dataset_0064/train_baseline.sh

# STEP 3B: train the iterative finetuning models
bash exp_scripts/dataset_0064/train_iterative_finetuning.sh 025
bash exp_scripts/dataset_0064/train_iterative_finetuning.sh 050
bash exp_scripts/dataset_0064/train_iterative_finetuning.sh 075
bash exp_scripts/dataset_0064/train_iterative_finetuning.sh 100

# STEP 3C: train the iterative finetuning models with correction
bash exp_scripts/dataset_0064/train_iterative_finetuning_with_correction.sh 025
bash exp_scripts/dataset_0064/train_iterative_finetuning_with_correction.sh 050
bash exp_scripts/dataset_0064/train_iterative_finetuning_with_correction.sh 075
bash exp_scripts/dataset_0064/train_iterative_finetuning_with_correction.sh 100

# STEP 4: we can graph our results; to see intermediate results, this script can be run 
# while the above 9 scripts are still running
python exp_scripts/generate_graphs.py 0064
```

#### $n = 64$, synthesizing motions using those trained weights

These scripts randomly select prompts from the test split for visualization, then sample from the
checkpoint, and render them. The second step takes a while, but note that you can execute the same
script $m$ times, where $m$ is the number of checkpoints that the script needs to sample from.

```bash
# STEP 1: run this script to copy over the relevant checkpoints into a new folder.
# command line arg #1: dataset size
# command line arg #2: quantity of prompts to sample from the test split
python exp_scripts/prep_for_visualization.py 0064 16

# STEP 2: sample motions from checkpoints, then render motions.
# command line arg #1: the path output from previous script
# command line arg #2: quantity of samples to synthesize for each prompt
python sample/checkpoint_visual_sampler.py exp_outputs/dataset_0064/visualization 4
```

</details>



<details style="margin-left: 20px;"><summary><b>dataset size = 128</b></summary>

#### $n = 128$, training from scratch

The logic for the case where the dataset has size $n=128$ is similar to the $n=64$ case; 
see above for a detailed description of what all these scripts are doing.

```bash
# train generation 0, then use it to seed other results
bash exp_scripts/dataset_0128/train_generation_0.sh
python exp_scripts/dataset_0128/copy_generation_0.py

# train generations 1 through 50
bash exp_scripts/dataset_0128/train_baseline.sh
bash exp_scripts/dataset_0128/train_iterative_finetuning.sh 025
bash exp_scripts/dataset_0128/train_iterative_finetuning.sh 050
bash exp_scripts/dataset_0128/train_iterative_finetuning.sh 075
bash exp_scripts/dataset_0128/train_iterative_finetuning.sh 100
bash exp_scripts/dataset_0128/train_iterative_finetuning_with_correction.sh 025
bash exp_scripts/dataset_0128/train_iterative_finetuning_with_correction.sh 050
bash exp_scripts/dataset_0128/train_iterative_finetuning_with_correction.sh 075
bash exp_scripts/dataset_0128/train_iterative_finetuning_with_correction.sh 100

# graph the results
python exp_scripts/generate_graphs.py 0128
```

#### $n = 128$, synthesizing motions using those trained weights


```bash
# copy the checkpoints into a new folder, randomly choose 16 prompts from test split
python exp_scripts/prep_for_visualization.py 0128 16

# synthesize motions from checkpoints, then render 4 samples for each one
python sample/checkpoint_visual_sampler.py exp_outputs/dataset_0128/visualization 4
```

</details>


<details style="margin-left: 20px;"><summary><b>dataset size = 256</b></summary>

#### $n = 256$, training from scratch

The logic for the case where the dataset has size $n=256$ is similar to the $n=64$ case; 
see above for a detailed description of what all these scripts are doing.

```bash
# train generation 0, then use it to seed other results
bash exp_scripts/dataset_0256/train_generation_0.sh
python exp_scripts/dataset_0256/copy_generation_0.py

# train generations 1 through 50
bash exp_scripts/dataset_0256/train_baseline.sh
bash exp_scripts/dataset_0256/train_iterative_finetuning.sh 025
bash exp_scripts/dataset_0256/train_iterative_finetuning.sh 050
bash exp_scripts/dataset_0256/train_iterative_finetuning.sh 075
bash exp_scripts/dataset_0256/train_iterative_finetuning.sh 100
bash exp_scripts/dataset_0256/train_iterative_finetuning_with_correction.sh 025
bash exp_scripts/dataset_0256/train_iterative_finetuning_with_correction.sh 050
bash exp_scripts/dataset_0256/train_iterative_finetuning_with_correction.sh 075
bash exp_scripts/dataset_0256/train_iterative_finetuning_with_correction.sh 100

# graph the results
python exp_scripts/generate_graphs.py 0256
```

#### $n = 256$, synthesizing motions using those trained weights


```bash
# copy the checkpoints into a new folder, randomly choose 16 prompts from test split
python exp_scripts/prep_for_visualization.py 0256 16

# synthesize motions from checkpoints, then render 4 samples for each one
python sample/checkpoint_visual_sampler.py exp_outputs/dataset_0256/visualization 4
```

</details>


<details style="margin-left: 20px;"><summary><b>dataset size = 2794</b></summary>


#### $n = 2794$, training from scratch

The logic for the case where the dataset has size $n=2794$ is similar to the $n=64$ case; 
see above for a detailed description of what all these scripts are doing.

```bash
# train generation 0, then use it to seed other results
bash exp_scripts/dataset_2794/train_generation_0.sh
python exp_scripts/dataset_2794/copy_generation_0.py

# train generations 1 through 50
bash exp_scripts/dataset_2794/train_baseline.sh
bash exp_scripts/dataset_2794/train_iterative_finetuning.sh 025
bash exp_scripts/dataset_2794/train_iterative_finetuning.sh 050
bash exp_scripts/dataset_2794/train_iterative_finetuning.sh 075
bash exp_scripts/dataset_2794/train_iterative_finetuning.sh 100
bash exp_scripts/dataset_2794/train_iterative_finetuning_with_correction.sh 025
bash exp_scripts/dataset_2794/train_iterative_finetuning_with_correction.sh 050
bash exp_scripts/dataset_2794/train_iterative_finetuning_with_correction.sh 075
bash exp_scripts/dataset_2794/train_iterative_finetuning_with_correction.sh 100

# graph the results
python exp_scripts/generate_graphs.py 2794
```

#### $n = 2794$, synthesizing motions using those trained weights


```bash
# copy the checkpoints into a new folder, randomly choose 16 prompts from test split
python exp_scripts/prep_for_visualization.py 2794 16

# synthesize motions from checkpoints, then render 4 samples for each one
python sample/checkpoint_visual_sampler.py exp_outputs/dataset_2794/visualization 4
```

</details>


</details>


<details>
  <summary><b>Synthesizing and rendering human motions using our pretrained weights</b></summary>
  

#### dataset size: $n = 64$

```bash
## in progress; we're uploading our weights soon!
```

</details>


## Acknowledgments

We thank the authors of the works we build upon:
- [Human Motion Diffusion Model](https://github.com/GuyTevet/motion-diffusion-model)
- [Universal Humanoid Controller](https://github.com/ZhengyiLuo/UHC)
- [VPoser aka Human Body Prior](https://github.com/nghorbani/human_body_prior)
- [Body Visualizer](https://github.com/nghorbani/body_visualizer)
- [HumanML3D](https://github.com/EricGuo5513/HumanML3D)
- [Conditional_Diffusion_MNIST](https://github.com/TeaPearce/Conditional_Diffusion_MNIST)

Our visualizations are inspired by
- [PhysDiff: Physics-Guided Human Motion Diffusion Model](https://nvlabs.github.io/PhysDiff/)
- [RoHM: Robust Human Motion Reconstruction via Diffusion](https://github.com/sanweiliti/RoHM)


## Bibtex
If you find this code useful in your research, please cite:

```
@misc{gillman2024selfcorrecting,
  title={Self-Correcting Self-Consuming Loops for Generative Model Training}, 
  author={Nate Gillman and Michael Freeman and Daksh Aggarwal and Chia-Hong Hsu and Calvin Luo and Yonglong Tian and Chen Sun},
  year={2024},
  eprint={2402.07087},
  archivePrefix={arXiv},
  primaryClass={cs.LG}
}
```
