''' 
This script does conditional image generation on MNIST, using a diffusion model

This code is modified from,
https://github.com/cloneofsimo/minDiffusion

Diffusion model is based on DDPM,
https://arxiv.org/abs/2006.11239

The conditioning idea is taken from 'Classifier-Free Diffusion Guidance',
https://arxiv.org/abs/2207.12598

This technique also features in ImageGen 'Photorealistic Text-to-Image Diffusion Modelswith Deep Language Understanding',
https://arxiv.org/abs/2205.11487

'''

from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image, make_grid
import numpy as np
import os
import copy
import argparse
import time
from sklearn.cluster import KMeans
from models import DDPM, ContextUnet
from fid_lenet import get_fid_score
import json
import shutil
import sys

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_type",
        choices=["baseline", "baseline_copy", "iterative_finetuning", "iterative_finetuning_with_correction"],
        type=str,
        required=True
    )
    parser.add_argument(
        "--synth_aug_percent",
        type=float,
        required=True
    )
    parser.add_argument(
        "--n_epoch_for_training_from_scratch",
        type=int,
        required=True
    )
    parser.add_argument(
        "--n_clusters_per_digit",
        type=int,
        required=False
    )
    parser.add_argument(
        "--fraction_of_train_set_to_train_on",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--save_dir_parent",
        type=str,
    )
    parser.add_argument(
        "--lr_divisor",
        type=float,
        default=1.0
    )
    parser.add_argument(
        "--resume_starting_at_generation",
        type=int,
        default=0
    )
    
    args = parser.parse_args()

    return args



def train_mnist():

    args = parse_args()
    print("args:", args)

    n_epoch_for_training_from_scratch = args.n_epoch_for_training_from_scratch
    batch_size = 256
    
    n_T = 400
    device = "cuda:0"
    n_classes = 10
    n_feat = 128
    lrate = 1e-4
    save_dir_parent = args.save_dir_parent
    train_type = args.train_type
    synth_aug_percent = args.synth_aug_percent
    if "baseline" in train_type:
        if args.fraction_of_train_set_to_train_on > 0.0:
            save_dir_original = os.path.join(save_dir_parent, train_type + "_" + str(args.fraction_of_train_set_to_train_on))
        else:
            save_dir_original = os.path.join(save_dir_parent, train_type + "_" + str(0.0))
    else:
        save_dir_original = os.path.join(save_dir_parent, train_type + "_" + str(synth_aug_percent))

    ws_test = [0.5] # strength of generative guidance
    n_generations = 50

    n_samples_to_generate_for_fid = 100


    ddpm = DDPM(nn_model=ContextUnet(in_channels=1, n_feat=n_feat, n_classes=n_classes), betas=(1e-4, 0.02), n_T=n_T, device=device, drop_prob=0.1)
    ddpm.to(device)

    tf = transforms.Compose([transforms.ToTensor()]) # mnist is already normalised 0 to 1

    dataset_original = MNIST("./dataset", train=True, download=True, transform=tf)
    dataset_original.data = dataset_original.data[:int(dataset_original.data.shape[0]*args.fraction_of_train_set_to_train_on)]
    dataset_original.targets = dataset_original.targets[:int(dataset_original.targets.shape[0]*args.fraction_of_train_set_to_train_on)]
    dataloader_original = DataLoader(dataset_original, batch_size=batch_size, shuffle=True, num_workers=0)

    optim = torch.optim.Adam(ddpm.parameters(), lr=lrate)


    if args.train_type == "iterative_finetuning_with_correction":
        # initialize the data needed to define the correction function

        gt_data = copy.deepcopy(dataset_original.data).numpy().reshape(len(dataset_original.data),-1) # (60000, 784)
        gt_context = dataset_original.targets.numpy()

        kmeans_per_digit, cluster_centers_rounded_per_digit = [], []

        centers_for_viz_all = []
        for digit in range(n_classes):

            idxs_for_this_digit = (gt_context == digit)
            gt_data_for_this_digit = gt_data[idxs_for_this_digit]

            # initialize kmeans
            kmeans = KMeans(n_clusters=args.n_clusters_per_digit, random_state=0, n_init="auto").fit(gt_data_for_this_digit)
            kmeans_per_digit.append(kmeans)
            cluster_centers_rounded = np.clip(np.round(0.5 + kmeans.cluster_centers_), 0, 255) # n_clusters_per_digit, 784
            cluster_centers_rounded_per_digit.append(cluster_centers_rounded)

            # visualize the clusters
            centers_for_viz = torch.tensor((cluster_centers_rounded/256)*-1 + 1).reshape(args.n_clusters_per_digit, 28, 28)
            centers_for_viz = centers_for_viz.unsqueeze(1)
            centers_for_viz_all.append(centers_for_viz)
        
        centers_for_viz_all = torch.concatenate(centers_for_viz_all)
        grid = make_grid(centers_for_viz_all, args.n_clusters_per_digit)
        os.makedirs(save_dir_original, exist_ok=True)
        save_image(grid, os.path.join(save_dir_original, f"digit_clusters.png"))



    save_dirs = []

    for generation in range(n_generations+1):

        save_dir = os.path.join(save_dir_original, "{:0>{}}".format(generation, 2))
        save_dirs.append(save_dir)
        
        if generation == 0 and args.resume_starting_at_generation == 0:

            print("\n"+"-"*50 + f"\nCommencing generation {generation}.\n" + "-"*50+"\n")

            if train_type == "baseline":

                os.makedirs(save_dir, exist_ok=True)
                # in this case, train from scratch for the quantity of epochs

                # train until close to convergence, i.e., 20ish epochs
                eval_metrics = {}
                for ep in range(n_epoch_for_training_from_scratch):

                    # TRAIN
                    print(f'Training epoch {ep}. Dataloader size: {len(dataloader_original)}')
                    ddpm.train()

                    # linear lrate decay
                    optim.param_groups[0]['lr'] = lrate*(1-ep/n_epoch_for_training_from_scratch)

                    pbar = tqdm(dataloader_original)
                    loss_ema = None
                    for x, c in pbar:
                        optim.zero_grad()
                        x = x.to(device)
                        c = c.to(device)
                        loss = ddpm(x, c)
                        loss.backward()
                        if loss_ema is None:
                            loss_ema = loss.item()
                        else:
                            loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                        pbar.set_description(f"loss: {loss_ema:.4f}")
                        optim.step()
                    
                    # EVAL
                    print(f'Evaluating epoch {ep}')
                    # save image of currently generated samples (top four rows)]
                    ddpm.eval()
                    with torch.no_grad():
                        n_sample = 4*n_classes
                        for w_i, w in enumerate(ws_test):
                            context, x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)

                            grid = make_grid(x_gen*-1 + 1, nrow=10)
                            os.makedirs(save_dir, exist_ok=True)
                            save_image_dir = os.path.join(save_dir, f"image_ep{ep}_w{w}.png")
                            save_image(grid, save_image_dir)
                            print('saved image at ' + save_image_dir)
                    
                    # compute + save metrics
                    fid_score, diversity_score = get_fid_score(ddpm, ws_test[0], n_samples_to_generate=n_samples_to_generate_for_fid)
                    # dump the metrics into the eval_dict json
                    eval_dict_path = os.path.join(save_dir, "eval_dict.json")
                    try:
                        with open(eval_dict_path, "r") as f:
                            eval_metrics = json.load(f)
                    except IOError:
                        print(f"initializing {eval_dict_path}")
                    eval_metrics[str(ep)] = {"0" : {"FID" : fid_score, "Diversity" : diversity_score}}
                    with open(eval_dict_path, "w") as fp:
                        json.dump(eval_metrics, fp, indent=4)
                    
                    # save model
                    if ep == int(n_epoch_for_training_from_scratch-1):
                        save_model_path = os.path.join(save_dir, f"model_{ep}.pth")
                        torch.save(ddpm.state_dict(), save_model_path)
                        print('saved model at ' + save_model_path)
            elif train_type in ["iterative_finetuning", "iterative_finetuning_with_correction", "baseline_copy"]:
                # in this case, seed the experiment using the baseline
                baseline_gen_00_dir = os.path.join(save_dir_parent, "baseline_0.2/00")
                shutil.copytree(baseline_gen_00_dir, save_dir)

        elif generation > 0 and generation >= args.resume_starting_at_generation :

            print("\n" + "-"*50 + f"\nCommencing generation {generation}.\n" + "-"*50+"\n")

            os.makedirs(save_dir, exist_ok=True)

            # STEP 1: load most recent model from previous generation
            prev_model_name = [fname for fname in os.listdir(save_dirs[-2]) if fname.endswith(".pth")][0]
            prev_model_path = os.path.join(save_dirs[-2], prev_model_name)
            print(f"Loading model from: {prev_model_path}")
            ddpm.load_state_dict(torch.load(prev_model_path))

            # STEP 2: synthesize new images using that model

            # STEP 2a: synthesize the motions
            if train_type in ["iterative_finetuning", "iterative_finetuning_with_correction"]:
                ddpm.eval()
                with torch.no_grad():
                    n_sample = int(np.ceil(synth_aug_percent * len(dataset_original)))
                    w = ws_test[0]

                    # synthesize new images
                    start_time = time.time()
                    x_gen_contexts, x_gens = [], []
                    samples_per_batch = 40 # must be a multiple of n_classes=10, chosen to optimize run time
                    n_batches = int(np.ceil(n_sample / samples_per_batch))
                    for batch in range(n_batches):
                        print(f"    synthesizing image batch {batch+1}/{n_batches}...")
                        x_gen_context, x_gen, x_gen_store = ddpm.sample(samples_per_batch, (1, 28, 28), device, guide_w=w)
                        x_gen_contexts.append(x_gen_context)
                        x_gens.append(x_gen)
                        
                    x_gen_context = torch.cat(x_gen_contexts)[:n_sample]
                    x_gen = torch.cat(x_gens)[:n_sample]
                    print(f"Finished synthesizing {n_sample} samples, total duration {time.time()-start_time}s")

                    # deepcopy dataset original, append data + labels to end of this; note it's normalized from 0 to 255, with dtype uint8
                    x_gen_normalized = x_gen[:,0].mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8) # 60, 28, 28


                # STEP 2b: apply self-correction operation
                if train_type == "iterative_finetuning_with_correction":
                
                    x_gens_context, x_gens_normalized = [], []
                    for digit in range(n_classes):

                        synth_data_for_this_digit = x_gen_normalized[(x_gen_context == digit).to("cpu")]
                        predicted_centers = kmeans_per_digit[digit].predict(synth_data_for_this_digit.reshape(synth_data_for_this_digit.shape[0],-1)) # n_samples/n_classes
                        cluster_centers_rounded = np.clip(np.round(0.5 + kmeans.cluster_centers_), 0, 255) # n_clusters_per_digit, 784

                        corrected_data_for_this_digit = cluster_centers_rounded_per_digit[digit][predicted_centers] # n_samples/n_classes, 784

                        x_gen_normalized_this_digit = torch.tensor(corrected_data_for_this_digit, dtype=torch.uint8).reshape(corrected_data_for_this_digit.shape[0], 28, 28)
                        x_gens_normalized.append(x_gen_normalized_this_digit)
                        x_gens_context.append(torch.tensor([digit] * x_gen_normalized_this_digit.shape[0]))
                    
                    generated_and_corrected_images = torch.cat(x_gens_normalized)
                    labels_for_generated_and_corrected_images = torch.cat(x_gens_context)

            # STEP 3: alter the dataloader to incorporate those examples
            if train_type in ["baseline", "baseline_copy"]:
                # no augmentation in this case
                dataset_augmented = copy.deepcopy(dataset_original)
                if args.fraction_of_train_set_to_train_on > 0.0:
                    dataset_augmented.data = dataset_augmented.data[:int(dataset_augmented.data.shape[0]*args.fraction_of_train_set_to_train_on)]
                    dataset_augmented.targets = dataset_augmented.targets[:int(dataset_augmented.targets.shape[0]*args.fraction_of_train_set_to_train_on)]
                dataloader_augmented = DataLoader(dataset_original, batch_size=batch_size, shuffle=True, num_workers=0)
            elif train_type in ["iterative_finetuning"]:
                dataset_augmented = copy.deepcopy(dataset_original)
                dataset_augmented.targets = torch.cat((dataset_augmented.targets, x_gen_context.to("cpu")), 0)
                dataset_augmented.data = torch.cat((dataset_augmented.data, x_gen_normalized), 0)
                dataloader_augmented = DataLoader(dataset_augmented, batch_size=batch_size, shuffle=True, num_workers=0)
            elif train_type == "iterative_finetuning_with_correction":
                dataset_augmented = copy.deepcopy(dataset_original)
                dataset_augmented.data = torch.cat((dataset_augmented.data, generated_and_corrected_images), 0)
                dataset_augmented.targets = torch.cat((dataset_augmented.targets, labels_for_generated_and_corrected_images), 0)
                dataloader_augmented = DataLoader(dataset_augmented, batch_size=batch_size, shuffle=True, num_workers=0)

                

            # STEP 4: fine-tune the model using those examples

            # TRAIN
            print(f'Training generation ' + "{:0>{}}".format(generation, 2) + f" using {train_type}. Dataloader size: {len(dataloader_augmented)}")

            # linear lrate decay
            optim.param_groups[0]['lr'] = lrate / (args.lr_divisor * n_epoch_for_training_from_scratch)

            pbar = tqdm(dataloader_augmented)
            loss_ema = None
            for x, c in pbar:
                optim.zero_grad()
                x = x.to(device)
                c = c.to(device)
                loss = ddpm(x, c)
                loss.backward()
                if loss_ema is None:
                    loss_ema = loss.item()
                else:
                    loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
                pbar.set_description(f"loss: {loss_ema:.4f}")
                optim.step()
            
            # EVAL
            print(f'Evaluating generation ' + "{:0>{}}".format(generation, 2) + f" using {train_type}")
            # save image of currently generated samples (four rows)
            ddpm.eval()
            with torch.no_grad():
                n_sample = 4*n_classes
                for w_i, w in enumerate(ws_test):
                    context, x_gen, x_gen_store = ddpm.sample(n_sample, (1, 28, 28), device, guide_w=w)

                    grid = make_grid(x_gen*-1 + 1, nrow=10)
                    save_image_dir = os.path.join(save_dir, f"image_gen{generation}_w{w}.png")
                    save_image(grid, save_image_dir)
                    print('saved image at ' + save_image_dir)

            # compute + save metrics
            fid_score, diversity_score = get_fid_score(ddpm, ws_test[0], n_samples_to_generate=n_samples_to_generate_for_fid)
            # dump the metrics into the eval_dict json
            eval_dict_path = os.path.join(save_dir, "eval_dict.json")
            eval_metrics = {"0" : {"FID" : fid_score, "Diversity" : diversity_score}}
            with open(eval_dict_path, "w") as fp:
                json.dump(eval_metrics, fp, indent=4)

            # save model
            save_model_path = os.path.join(save_dir, f"model_gen{generation}.pth")
            torch.save(ddpm.state_dict(), save_model_path)
            print('saved model at ' + save_model_path)
            sys.stdout.flush()





if __name__ == "__main__":
    train_mnist()

