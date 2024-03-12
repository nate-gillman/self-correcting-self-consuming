import shutil
import os
import json


def copy_files(source_dir, target_dir):
    # List of files to copy
    files_to_copy = ['model000010000.pt', 'opt000010000.pt', 'eval_dict.json', 'args.json']
    specific_key = files_to_copy[0]

    # Ensure target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Copy each specified file from the source to the target directory
    for file_name in files_to_copy:
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, file_name)

        # Check if the source file exists before attempting to copy
        if os.path.exists(source_file):
            shutil.copy2(source_file, target_file)
            print(f'Copied {file_name} to {target_dir}')
        else:
            print(f'File {file_name} does not exist in the source directory.')

    # Special handling for eval_dict.json to copy only the specified subset
    eval_dict_path = os.path.join(source_dir, 'eval_dict.json')
    if os.path.exists(eval_dict_path):
        with open(eval_dict_path, 'r') as file:
            eval_dict = json.load(file)

        # Extract the specific key-value pair
        specific_eval_dict = {specific_key: eval_dict.get(specific_key)}

        # Write the specific subset to the target directory
        target_eval_dict_path = os.path.join(target_dir, 'eval_dict.json')
        with open(target_eval_dict_path, 'w') as file:
            json.dump(specific_eval_dict, file, indent=4)
        print(f'Copied specific key-value pair from eval_dict.json to {target_dir}')
    else:
        print('File eval_dict.json does not exist in the source directory.')


def main():
    source_directory = 'exp_outputs/dataset_0128/generation_0/generation_0'

    # first, baseline
    target_directory = f'exp_outputs/dataset_0128/baseline/generation_0'
    copy_files(source_directory, target_directory)

    # then, iterative_finetuning experiments
    for augmentation_percentage in ["025", "050", "075", "100"]:
        for experiment_type in ["iterative_finetuning", "iterative_finetuning_with_correction"]:
            target_directory = f'exp_outputs/dataset_0128/synthetic_percent_{augmentation_percentage}_{experiment_type}/generation_0'
            copy_files(source_directory, target_directory)


if __name__ == "__main__":
    main()
