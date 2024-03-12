#!/bin/bash
set -e

# move to the location where this script exists, it should be in the root of the SCSC repository.
# assuming this script is /utils/setup/get_smpl_data.sh, repo_root should be set to the root level directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
cd ../../
repo_root=$(pwd)
echo "extract_humanml3d.sh: repository root folder appears to be: $repo_root"

# assumes that the runner has used: the AMASS/HumanML3D Downloader script
# located in: utils/repository/
# move all data to here
extract_dir="amass_data"

mkdir -p "$extract_dir"

files=(
  "ACCD"
  "MPI_HDM05"
  "TCD_handMocap"
  "SFU"
  "BMLmovi"
  "CMU"
  "MPI_mosh"
  "EKUT"
  "KIT"
  "Eyes_Janpan_Dataset"
  "BMLhandball"
  "Transitions_mocap"
  "MPI_Limits"
  "HumanEva"
  "SSM_synced"
  "DFaust_67"
  "TotalCapture"
  "BioMotionLab_NTroje"
)

for file in "${files[@]}"; do
    # Extract the contents of the file into the destination directory
    tar -xzvf "$file.tar.bz2" -C "$extract_dir"
    rm "$file.tar.bz2"
done
