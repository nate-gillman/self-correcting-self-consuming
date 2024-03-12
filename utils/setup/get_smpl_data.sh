#!/bin/bash
set -e

# set the source to the conda env
conda_sh_location="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
source "$conda_sh_location"
conda activate scsc

# move to the location where this script exists, it should be in the root of the SCSC repository.
# assuming this script is /utils/setup/get_smpl_data.sh, repo_root should be set to the root level directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
cd ../../
repo_root=$(pwd)
echo "get_smpl_data.sh: repository root folder appears to be: $repo_root"

# run interactive script to download SMPL data
python utils/repository/download_smpl.py

# VPOSER
if [ -f "$repo_root/V02_05.zip" ]; then
  unzip -o V02_05.zip -d "$repo_root/body_models"
  mv "$repo_root/body_models/V02_05" "$repo_root/body_models/vposer_v2_05"
  rm V02_05.zip
fi

# SMPL
if [ -f "$repo_root/SMPL_python_v.1.1.0.zip" ]; then
  unzip -o SMPL_python_v.1.1.0.zip "SMPL_python_v.1.1.0/smpl/models/*" -d "$repo_root/"
  mv SMPL_python_v.1.1.0/smpl/models/basicmodel_f_lbs_10_207_0_v1.1.0.pkl body_models/smpl/SMPL_FEMALE.pkl
  chmod u+x body_models/smpl/SMPL_FEMALE.pkl
  mv SMPL_python_v.1.1.0/smpl/models/basicmodel_m_lbs_10_207_0_v1.1.0.pkl body_models/smpl/SMPL_MALE.pkl
  chmod u+x body_models/smpl/SMPL_MALE.pkl
  rm -rf SMPL_python_v.1.1.0
  rm SMPL_python_v.1.1.0.zip
fi

# DMPLS
if [ -f "$repo_root/dmpls.tar.xz" ]; then
  mkdir -p body_models/dmpls
  tar -xvf dmpls.tar.xz -C "$repo_root/body_models/dmpls"
  rm dmpls.tar.xz
fi

# SMPLH
if [ -f "$repo_root/smplh.tar.xz" ]; then
  mkdir -p body_models/smplh
  tar -xvf smplh.tar.xz -C "$repo_root/body_models/smplh"
  rm smplh.tar.xz
fi

# AMASS/HumanML3D
echo "For processing AMASS data for use in HumanML3D, please run: utils/setup/extract_humanml3d.sh."