import os
import shutil
import random
import sys

def aggregate_data(src_path, checkpoint_number, num_prompts=4, dataset_size="0064"):
    # Define the destination path
    dst_path = os.path.join(src_path, 'visualization')
    
    # Create directories if they do not exist
    dirs = {
        'checkpoints': os.path.join(dst_path, 'checkpoints'),
        'motions_gt': os.path.join(dst_path, 'motions_gt'),
        'texts': os.path.join(dst_path, 'texts'),
    }
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    if dataset_size in ["0064", "0128", "0256"]:
        base_paths = [
            'baseline',
            'synthetic_percent_025_iterative_finetuning',
            'synthetic_percent_050_iterative_finetuning',
            'synthetic_percent_075_iterative_finetuning',
            'synthetic_percent_100_iterative_finetuning',
            'synthetic_percent_025_iterative_finetuning_with_correction',
            'synthetic_percent_050_iterative_finetuning_with_correction',
            'synthetic_percent_075_iterative_finetuning_with_correction',
            'synthetic_percent_100_iterative_finetuning_with_correction',
        ]
    elif dataset_size in ["2794"]:
        base_paths = [
            'baseline',
            'synthetic_percent_05_iterative_finetuning',
            'synthetic_percent_10_iterative_finetuning',
            'synthetic_percent_15_iterative_finetuning',
            'synthetic_percent_20_iterative_finetuning',
            'synthetic_percent_25_iterative_finetuning',
            'synthetic_percent_05_iterative_finetuning_with_correction',
            'synthetic_percent_10_iterative_finetuning_with_correction',
            'synthetic_percent_15_iterative_finetuning_with_correction',
            'synthetic_percent_20_iterative_finetuning_with_correction',
            'synthetic_percent_25_iterative_finetuning_with_correction',
        ]
    else:
        raise NotImplementedError

    # Copy model files to checkpoints
    for base_path in base_paths:

        # where we'll be copying data from
        model_src_dir = os.path.join(src_path, base_path, f'generation_{checkpoint_number}')

        # directory where well be copying data to
        base_path_subdir = os.path.join(dirs['checkpoints'], f"generation_{checkpoint_number}_{base_path}")
        os.makedirs(base_path_subdir, exist_ok=True)

        # copy the model and opt files over
        for file in os.listdir(model_src_dir):
            # copy + rename the model file
            if file.startswith('model'):
                shutil.copy(os.path.join(model_src_dir, file), os.path.join(base_path_subdir, "model.pt"))
            # copy + rename the opt file
            if file.startswith('opt'):
                shutil.copy(os.path.join(model_src_dir, file), os.path.join(base_path_subdir, "opt.pt"))

        # copy the args.json file over
        for file in os.listdir(os.path.dirname(model_src_dir)):
            if file.startswith('args'):
                shutil.copy(os.path.join(os.path.dirname(model_src_dir), file), os.path.join(base_path_subdir, "args.json"))
                break            

    # Randomly choose and copy text files
    test_file_path = os.path.join('dataset', f'HumanML3D_subset_{dataset_size}', 'test.txt')
    with open(test_file_path, 'r') as file:
        lines = file.readlines()
    selected_elements = random.sample(lines, num_prompts)
    for element in selected_elements:
        element = element.strip()
        src_txt_file = os.path.join('dataset', f'HumanML3D_subset_{dataset_size}', 'texts', f'{element}.txt')
        shutil.copy(src_txt_file, dirs['texts'])

    # Copy .npy files to motions_gt
    motions_src_dir = os.path.join('dataset', f'HumanML3D_subset_{dataset_size}', 'new_joint_vecs')
    for element in selected_elements:
        element = element.strip()
        src_npy_file = os.path.join(motions_src_dir, f'{element}.npy')
        shutil.copy(src_npy_file, dirs['motions_gt'])

if __name__ == "__main__":
    
    dataset_size = str(sys.argv[1]) # e.g. "0064"
    num_prompts = int(sys.argv[2])  # e.g. 16

    # Example usage
    src_path = f"exp_outputs/dataset_{dataset_size}"
    checkpoint_number = 50

    aggregate_data(src_path, checkpoint_number, num_prompts=num_prompts, dataset_size=dataset_size)
