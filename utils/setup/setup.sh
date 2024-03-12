#!/bin/bash
#
# The purpose of this script is to setup the SCSC environment.
# Unfortunately, setting this up is quite involved. Some of the 'gotchas' and quirks to getting
# it to work together are captured below.
#
# We aim to not edit any of the host's global settings or files in this package.
#
# There are 3 particularly difficult dependencies to install here, which are:
#   1. the Boost library, which we need to compile some libraries that have non-python code
#   2. OpenGL and its related libraries
#   3. Mujoco, specifically cythonizing Mujoco-py. This is harder on Mac OS than Linux. May require special
#      care that this script cannot cover.
#

# add some constants, source: https://stackoverflow.com/a/5947802/23160579
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m'

set -e

is_linux=false
os_name=$(uname -s)
if [ "$os_name" == "Darwin" ]; then
  echo -e "${YELLOW}Warning, this environment setup script supports Mac OS but training requires Linux.${NC}"
  sleep 1
elif [ "$os_name" == "Linux" ]; then
  is_linux=true
else
  echo -e "${YELLOW}Apologies, this setup script does not support Windows.${NC}"
  exit 1
fi

# enable the parent script (this one) to fail fast if any children are given a SIGKILL or SIGINT
function exit_script() {
    echo "Cancelling SCSC setup."
    exit 1
}
trap exit_script INT TERM

# move to the location where this script exists, it should be in the root of the SCSC repository.
# assuming this script is /utils/setup/setup.sh, repo_root should be set to the root level directory
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
cd ../../
repo_root=$(pwd)
echo "setup.sh: repository root folder appears to be: $repo_root"

# set the source to the conda env
conda_sh_location="$(dirname "$(dirname "$CONDA_EXE")")/etc/profile.d/conda.sh"
source "$conda_sh_location"

# remove conda environment if it exists
conda deactivate || true

# --- REMOVE THE EXISTING ENVIRONMENT IF THERE IS ONE ---
echo -e "${RED}Removing existing SCSC conda environment. Will ask for confirmation...${NC}"
conda remove -n scsc --all

# create the conda environment
# Why Python 3.8.12? See here:
#   https://github.com/NixOS/nixpkgs/issues/105038#issuecomment-912119777
conda create -n scsc python==3.8.12 --yes

echo -e "${RED}Activating new SCSC conda environment.${NC}"
conda activate scsc

# install core dependencies
if [ "$is_linux" = true ]; then
  conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -c conda-forge --yes
else
  # cudatoolkit does not support Mac OS past v. 9.0. Try to install the latest version we can find
  conda install cudatoolkit --yes
  conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch -c conda-forge --yes
fi

conda install libgcc --yes
conda install -c conda-forge spacy --yes
conda install -c conda-forge boost --yes

echo -e "${RED}Done with core conda setup.${NC}"

# install MDM dependencies
pip install git+https://github.com/openai/CLIP.git
pip install wandb
pip install smplx
pip install scipy
pip install chumpy

# install UHC dependencies
# -- install mujoco
# May need to troubleshoot on Mac OS: https://github.com/openai/mujoco-py?tab=readme-ov-file#troubleshooting
if [ "$is_linux" = true ]; then
  # Linux
  wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
  tar -xzf mujoco210-linux-x86_64.tar.gz
  mv -n mujoco210 mujoco
  rm mujoco210-linux-x86_64.tar.gz
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/mujoco/bin
else
  # Mac OS
  llvm_folder="/usr/local/opt/llvm/bin"
  if [ -d "$llvm_folder" ]; then
      # from: https://github.com/openai/mujoco-py?tab=readme-ov-file#youre-on-macos-and-you-see-clang-error-unsupported-option--fopenmp
      echo "Found LLVM folder. Setting environment variables..."
      export PATH="$llvm_folder:$PATH"
      export CC="$llvm_folder/clang"
      export CXX="$llvm_folder/clang++"
      export CXX11="$llvm_folder/clang++"
      export CXX14="$llvm_folder/clang++"
      export CXX17="$llvm_folder/clang++"
      export CXX1X="$llvm_folder/clang++"
      export LDFLAGS="-L/usr/local/opt/llvm/lib"
      export CPPFLAGS="-I/usr/local/opt/llvm/include"
      echo "Environment variables set."
  else
      echo "LLVM folder not found. No changes made."
  fi

  # download the mujoco files
  wget https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-macos-x86_64.tar.gz
  tar -xzf mujoco210-macos-x86_64.tar.gz
  mv -n mujoco210 mujoco
  rm mujoco210-macos-x86_64.tar.gz
fi

# set the mujoco path to locally downloaded
conda env config vars set MUJOCO_PY_MUJOCO_PATH="$(PWD)"/mujoco
MUJOCO_PY_MUJOCO_PATH=$PWD/mujoco
export MUJOCO_PY_MUJOCO_PATH

# -- install python dependencies
pip install "mujoco-py<2.2,>=2.1"
pip install "cython<3"
pip install gym

# install visualization dependencies
pip install pandas
pip install colour
pip install loguru
pip install omegaconf
pip install git+https://github.com/nghorbani/body_visualizer
pip install trimesh
pip install joblib

conda install scikit-image --yes

pip install lxml
pip install vtk
pip install numpy-stl
pip install ipdb
pip install blobfile

BOOST_INCLUDE_DIRS=$CONDA_PREFIX/include/boost make all -C translation/utils_mdm_to_amass/mesh-master/

# make it so the environment can find psbody from our locally compiled source, in a line like:
# from psbody.mesh import Mesh
PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/translation/utils_mdm_to_amass"
export PYTHONPATH
conda env config vars set PYTHONPATH="${PYTHONPATH}"

# see: https://github.com/ethz-asl/reinmav-gym/issues/35#issuecomment-1222946797
conda install -c conda-forge mesalib glew glfw --yes

# add this line to .bashrc, with your path inserted
export CPATH=$CONDA_PREFIX/include

if [ "$is_linux" = true ]; then
  # not supported on Mac OS
  # see: https://github.com/mayeut/patchelf-pypi?tab=readme-ov-file#platforms
  pip install patchelf
fi

conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge --yes
pip install numpy==1.23.1

# see: https://github.com/pydantic/pydantic/issues/545#issuecomment-1615031353
pip install pydantic==1.10.10
pip install pyrender

# see: https://github.com/EricGuo5513/text-to-motion/issues/23
pip install matplotlib==3.3.1

# get training data
pip install gdown

# needed for gaussian toy example
conda install -c conda-forge scikit-learn --yes

# needed for rendering
pip install PyOpenGL -U
pip install --force-reinstall imageio==2.23.0
pip install numpy==1.23.1

echo -e "${RED}Downloading data dependencies...${NC}"

# set up the minimal MDM data, we will create a symbolic link to it for UHC to use.
./prepare/download_smpl_files.sh
./prepare/download_glove.sh
./prepare/download_t2m_evaluators.sh

mkdir -p UniversalHumanoidControl/data
ln -s "$(realpath body_models/smpl/)" UniversalHumanoidControl/data/smpl

# setup UHC data
cd UniversalHumanoidControl
./download_data.sh
cd ../
echo -e "${RED}Running an import test on the environment...${NC}"

# if this runs without error the environment was properly setup.
python utils/setup/test_environment.py

echo -e "${GREEN}LGTM. Next, you will need to download MDM's data. See the /prepare folder.${NC}"
echo "The setup script has already run /prepare/download_smpl_files.sh, /prepare/download_glove.sh and /prepare/download_t2m_evaluators.sh".
echo -e "${GREEN}Activate the SCSC environment with: conda activate scsc.${NC}"

echo "The environment is finished setting up, but we still need the SMPL data."
echo "We must download the SMPL body models and SMPLX VPoser models."
read -r -p "Would you like to proceed? [Y/n] " answer

answer_lowercase=$(echo "$answer" | tr '[:upper:]' '[:lower:]')
if [[ "$answer_lowercase" == "y" || "$answer_lowercase" == "" ]]; then
    echo "User stated they wish to download SMPL/SMPLX data."
else
    echo "User stated they did not wish to download SMPL/SMPLX data."
    exit 0
fi

# run the smpl script
./utils/setup/get_smpl_data.sh