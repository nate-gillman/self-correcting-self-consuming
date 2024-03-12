"""
Train a diffusion model on images.

This code is based on https://github.com/openai/guided-diffusion
"""
import os
import shutil

from torch.utils.data import DataLoader

import wandb
from data_loaders.get_data import get_dataset_loader
from train.train_platforms import get_train_platform
from train.training_loop import SelfConsumingTrainLoop, TrainLoop
from utils import dist_util as distributed_training_util
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils.parser_util import (
    autophagous_train_args,
    get_args_save_path,
    get_json_of_args,
)
import torch
import json

from utils.generate_POC_samples_large_scale_MDM import (
    get_motions_idxs_and_captions_with_keyword, get_prompts_from
)
import random
from sample.generate import loop_generate
from utils.generate_POC_samples_large_scale_UHC_batch import uhc_correction_batch
import copy

def main() -> None:
    args = autophagous_train_args()


    if args.no_fixseed:
        print("NB: not fixing the random seed.")
    else:
        fixseed(args.seed)

    # configure the training platform
    train_platform = get_train_platform(args)

    # --- check if save_dir is specified and exists (make it if not) ---
    if args.save_dir is None:
        raise FileNotFoundError("save_dir was not specified.")
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError("save_dir [{}] already exists.".format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # dump the training args to file
    args_path = get_args_save_path(args)
    with open(args_path, "w") as fw:
        fw.writelines(get_json_of_args(args))

    # distributed_training_util.setup_dist(args.device)

    # --- load the train/test/eval data ---
    data: DataLoader = get_dataset_loader(
        name=args.dataset,
        batch_size=args.batch_size,
        # num_frames defaults to 60 and is the maximum number of frames to use in
        # training. If training with HumanML3D, this field is ignored
        num_frames=args.num_frames,
        subset_by_keyword=args.filter_dataset_by_keyword,
        synthetic_data_dir=args.synthetic_data_dir,
        synthetic_augmentation_percent=args.synthetic_augmentation_percent,
        augmentation_type=args.augmentation_type,
        nearest_neighbor_POC_type=args.nearest_neighbor_POC_type,
        mini_dataset_dir=args.mini_dataset_dir
    )

    # --- initialize models ---
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(distributed_training_util.dev())
    model.rot2xyz.smpl_model.eval()
    print(
        "Diffusion Model Total params: %.2fM"
        % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0)
    )

    # --- train the model ---
    if args.is_self_consuming:

        generation = args.start_at_generation

        print(f"INSIDE SELF CONSUMING LOOP; ABOUT TO BEGIN GENERATION {generation}")
        save_dir_original = copy.deepcopy(args.save_dir)
        
        # for first generation, just train from scratch
        if generation == 0:

            if args.use_wandb:
                project_name = args.wandb_project_name
                if not project_name:
                    raise RuntimeError(
                        "Argument set wandb is set to true, but no project name was given! "
                        "Set it with the argument --wandb_project_name"
                    )
                print("Using wandb...")
                # wandb.init(project=project_name, config=args.__dict__)
                if args.wandb_exp_name:
                    wandb_exp_name = f"{args.wandb_exp_name}-gen-0"
                else:
                    wandb_exp_name = None
                wandb.init(
                    project=project_name, 
                    config=args.__dict__, 
                    settings=wandb.Settings(start_method='fork'),
                    name=(wandb_exp_name or None)
                )
                # define our custom x axis metric
                wandb.define_metric("step")
                # define which metrics will be plotted against it
                wandb.define_metric("train/*", step_metric="batch_idx_train")
                wandb.define_metric("valid/*", step_metric="batch_idx_train")
                wandb.define_metric("epoch", step_metric="batch_idx_train")
                wandb.define_metric("learning_rate", step_metric="batch_idx_train")
                # and make it the default axis
                wandb.define_metric("*", step_metric="step")

            
            args.save_dir = os.path.join(save_dir_original, f"generation_{generation}")
            os.makedirs(args.save_dir, exist_ok=True)
            shutil.copy(os.path.join(save_dir_original, "args.json"), os.path.join(args.save_dir, "args.json"))

            TrainLoop(args, train_platform, model, diffusion, data).run_loop()
            if args.use_wandb:
                wandb.finish()

        # for subsequent generations, self-consuming logic that depends on previous model
        elif generation > 0:

            # define new save dir for this next generation
            args.save_dir = os.path.join(save_dir_original, f"generation_{generation}")
            os.makedirs(args.save_dir, exist_ok=True)
            synthetic_motions_dir = os.path.join(args.save_dir, "synthetic_motions")

            # STEP 1: identify best performing model (FID) from previous generation
            save_dir_previous = os.path.join(save_dir_original, f"generation_{generation-1}")
            eval_dict_path = os.path.join(save_dir_previous, "eval_dict.json")
            with open(eval_dict_path, "r") as f:
                eval_dict = json.load(f)
            best_model = max(eval_dict, key=lambda k: k) # alphabetical... get last one!!
            best_model_path = os.path.join(save_dir_previous, best_model)
            print(f"GENERATION IS {generation}, WERE USING THE LAST THE BEST MODEL PATH {best_model_path}")

            # STEP 2: load that model, for use in synthesizing new examples, OR in resuming training
            state_dict = torch.load(best_model_path, map_location="cpu")
            load_model_wo_clip(model, state_dict)

            if args.ONLY_SYNTHESIZE_EXAMPLES:

                print("INSIDE args.ONLY_SYNTHESIZE_EXAMPLES; synthesizing examples")
                # STEP 3: synthesize new motions

                # STEP 3a: get all the prompts that we want to imitate
                keyword = ""  # i.e., filter by trivial keyword, so include everything...
                data_root = "dataset/HumanML3D"
                if args.mini_dataset_dir:
                    data_root = args.mini_dataset_dir
                idx_to_caption = get_motions_idxs_and_captions_with_keyword(keyword, data_root)
                prompts = get_prompts_from(idx_to_caption)
                num_to_sample = int(args.synthetic_augmentation_percent*len(prompts))
                random.seed(generation*args.seed)
                prompts_subset = random.sample(prompts, num_to_sample)
                print("Number of prompts we're sampling: ", len(prompts_subset))

                # STEP 3b: generate motions using the current model
                print("synthesizing new motions into synthetic_motions_dir:", synthetic_motions_dir)
                loop_generate(prompts_subset, model, diffusion, data, "dummy_path", synthetic_motions_dir, True)

                # STEP 3c: do UHC-correction
                if args.augmentation_type == "imitation_output":
                    uhc_correction_batch(
                        synthetic_motions_dir, RECOMPUTE_AND_OVERWRITE=True, VIS=False, 
                    )
                    print("imitated synthetic motions, inside:", synthetic_motions_dir)
                print("END OF args.ONLY_SYNTHESIZE_EXAMPLES; synthesizing examples")
                return None

            elif not args.ONLY_SYNTHESIZE_EXAMPLES:

                print("INSIDE not args.ONLY_SYNTHESIZE_EXAMPLES; time to train")
                # STEP 3d: alter the dataloader to incorporate those examples
                print("building a new dataloader that incorporates motions from:", synthetic_motions_dir)
                data: DataLoader = get_dataset_loader(
                    name=args.dataset,
                    batch_size=args.batch_size,
                    # num_frames defaults to 60 and is the maximum number of frames to use in
                    # training. If training with HumanML3D, this field is ignored
                    num_frames=args.num_frames,
                    subset_by_keyword=args.filter_dataset_by_keyword,
                    synthetic_data_dir=synthetic_motions_dir,
                    synthetic_augmentation_percent=args.synthetic_augmentation_percent,
                    augmentation_type=args.augmentation_type,
                    is_fully_synthetic=args.is_fully_synthetic,
                    mini_dataset_dir=args.mini_dataset_dir
                )

                # STEP 4: re-initialize wandb
                if args.use_wandb:
                    project_name = args.wandb_project_name
                    if not project_name:
                        raise RuntimeError(
                            "Argument set wandb is set to true, but no project name was given! "
                            "Set it with the argument --wandb_project_name"
                        )                        
                    print("Using wandb...")
                    if args.wandb_exp_name:
                        wandb_exp_name = f"{args.wandb_exp_name}-gen-{generation}"
                    else:
                        wandb_exp_name = None
                    wandb.init(
                        project=project_name, 
                        config=args.__dict__, 
                        name=(wandb_exp_name or None), 
                        settings=wandb.Settings(start_method='fork')
                    )
                    # define our custom x axis metric
                    wandb.define_metric("step")
                    # define which metrics will be plotted against it
                    wandb.define_metric("train/*", step_metric="batch_idx_train")
                    wandb.define_metric("valid/*", step_metric="batch_idx_train")
                    wandb.define_metric("epoch", step_metric="batch_idx_train")
                    wandb.define_metric("learning_rate", step_metric="batch_idx_train")
                    # and make it the default axis
                    wandb.define_metric("*", step_metric="step")

                # STEP 5: train this from-scratch initialized model
                TrainLoop(args, train_platform, model, diffusion, data).run_loop()
                print("FINISHING UP not args.ONLY_SYNTHESIZE_EXAMPLES; finished training")
                if args.use_wandb:
                    wandb.finish()
                return None

    train_platform.close()


if __name__ == "__main__":
    main()
