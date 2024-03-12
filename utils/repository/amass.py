import click
import requests

import re
from typing import Optional
from pathlib import Path

from utils.repository.common import get_login_session_id, download_and_save_file, USER_AGENT

# datasets required for HumanML3D are listed here:
#   https://github.com/EricGuo5513/HumanML3D/blob/main/raw_pose_processing.ipynb
HUMAN_ML3D_DATASET_NAMES = [
    ("ACCAD", "ACCD"),
    ("HDM05", "MPI_HDM05"),
    ("TCDHands", "TCD_handMocap"),
    ("SFU", "SFU"),
    ("BMLmovi", "BMLmovi"),
    ("CMU", "CMU"),
    ("MoSh", "MPI_mosh"),
    ("EKUT", "EKUT"),
    ("KIT", "KIT"),
    ("EyesJapanDataset", "Eyes_Janpan_Dataset"),
    ("BMLhandball", "BMLhandball"),
    ("Transitions", "Transitions_mocap"),
    ("PosePrior", "MPI_Limits"),
    ("HumanEva", "HumanEva"),
    ("SSM", "SSM_synced"),
    ("DFaust", "DFaust_67"),
    ("TotalCapture", "TotalCapture"),
    ("BMLrub", "BioMotionLab_NTroje")
]

# These are all for the SMPL H+G versions.
_DATASET_PATHS = {
    'ACCAD': {'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/ACCAD.tar.bz2',
              'licensename': 'licences/accad.html'},
    'BMLhandball': {
        'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/BMLhandball.tar.bz2',
        'licensename': 'licences/bmlhandball.html'},
    'BMLmovi': {'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/BMLmovi.tar.bz2',
                'licensename': 'licences/bmlmovi.html'},
    'BMLrub': {'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/BMLrub.tar.bz2',
               'licensename': 'licences/bmlrub.html'},
    'CMU': {'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/CMU.tar.bz2',
            'licensename': 'licences/cmu.html'},
    'CNRS': {'sfile': 'amass_per_dataset/smplx/gender_specific/mosh_results/CNRS.tar.bz2',
             'licensename': 'licences/cnrs.html'},
    'DFaust': {'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/DFaust.tar.bz2',
               'licensename': 'licences/dfaust.html'},
    'DanceDB': {'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/DanceDB.tar.bz2',
                'licensename': 'licences/dancedb.html'},
    'EKUT': {'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/EKUT.tar.bz2',
             'licensename': 'licences/ekut.html'},
    'EyesJapanDataset': {
        'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/EyesJapanDataset.tar.bz2',
        'licensename': 'licences/eyesjapandataset.html'},
    'GRAB': {'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/GRAB.tar.bz2',
             'licensename': 'licences/grab.html'},
    'HDM05': {'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/HDM05.tar.bz2',
              'licensename': 'licences/hdm05.html'},
    'HUMAN4D': {'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/HUMAN4D.tar.bz2',
                'licensename': 'licences/human4d.html'},
    'HumanEva': {
        'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/HumanEva.tar.bz2',
        'licensename': 'licences/humaneva.html'},
    'KIT': {'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/KIT.tar.bz2',
            'licensename': 'licences/kit.html'},
    'MoSh': {'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/MoSh.tar.bz2',
             'licensename': 'licences/mosh.html'},
    'PosePrior': {
        'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/PosePrior.tar.bz2',
        'licensename': 'licences/poseprior.html'},
    'SFU': {'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/SFU.tar.bz2',
            'licensename': 'licences/sfu.html'},
    'SOMA': {'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/SOMA.tar.bz2',
             'licensename': 'licences/soma.html'},
    'SSM': {'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/SSM.tar.bz2',
            'licensename': 'licences/ssm.html'},
    'TCDHands': {
        'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/TCDHands.tar.bz2',
        'licensename': 'licences/tcdhands.html'},
    'TotalCapture': {
        'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/TotalCapture.tar.bz2',
        'licensename': 'licences/totalcapture.html'},
    'Transitions': {
        'sfile': 'amass_per_dataset/smplh/gender_specific/mosh_results/Transitions.tar.bz2',
        'licensename': 'licences/transitions.html'},
    'WEIZMANN': {
        'sfile': 'amass_per_dataset/smplx/gender_specific/mosh_results/WEIZMANN.tar.bz2',
        'licensename': 'licences/weizmann.html'}
}


def get_amass_session_id(username: str, password: str) -> str:
    host = "amass.is.tue.mpg.de"
    return get_login_session_id(host, username, password)


def get_has_agreed_to_license(dataset_name: str, amass_session_id: str) -> bool:
    filename = _DATASET_PATHS[dataset_name]['sfile']
    licensename = _DATASET_PATHS[dataset_name]['licensename']

    headers = {
        'Accept': 'text/plain, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': 'https://amass.is.tue.mpg.de/download.php',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': USER_AGENT,
        'X-Requested-With': 'XMLHttpRequest',
        'sec-ch-ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
    }

    response = requests.get(
        'https://amass.is.tue.mpg.de/admin/ajax_getlicenseagreed.php',
        params={
            'filename': filename,
            'licensename': licensename
        },
        cookies={
            'PHPSESSID': amass_session_id,
        },
        headers=headers,
    )

    return response.text == "1"


def get_license_text(licensename: str) -> str:
    headers = {
        'Referer': 'https://amass.is.tue.mpg.de/',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': USER_AGENT,
        'sec-ch-ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
    }
    response = requests.get(f'https://download.is.tue.mpg.de/amass/{licensename}', headers=headers)
    html = response.text

    # remove HTML tags
    text = re.sub('<[^<]+?>', '', html)

    # format for CLI output
    return text.replace("\t", "").strip()


def set_has_agreed_to_license(dataset_name: str, amass_session_id: str) -> None:
    filename = _DATASET_PATHS[dataset_name]['sfile']
    licensename = _DATASET_PATHS[dataset_name]['licensename']
    headers = {
        'Accept': 'text/plain, */*; q=0.01',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
        'Referer': 'https://amass.is.tue.mpg.de/download.php',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'same-origin',
        'User-Agent': USER_AGENT,
        'X-Requested-With': 'XMLHttpRequest',
        'sec-ch-ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
    }

    response = requests.get(
        'https://amass.is.tue.mpg.de/admin/ajax_setlicenseagreed.php',
        params={
            'filename': filename,
            'licensename': licensename
        },
        cookies={
            'PHPSESSID': amass_session_id,
        },
        headers=headers,
    )

    r = response.text

    if r != "1":
        raise RuntimeError("Failed to agree to license.")


def download_amass_file_smpl_h_g(amass_session_id: str, dataset_name: str, save_to: Path,
                                 rename_to: Optional[Path] = None) -> Optional[Path]:
    url = 'https://download.is.tue.mpg.de/download.php'
    headers = {
        'Host': 'download.is.tue.mpg.de',
        'Connection': 'keep-alive',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"macOS"',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,'
                  'image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Sec-Fetch-Site': 'same-site',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-User': '?1',
        'Sec-Fetch-Dest': 'document',
        'Referer': 'https://amass.is.tue.mpg.de/',
        'Accept-Language': 'en-US,en;q=0.9',
    }

    if dataset_name not in _DATASET_PATHS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Not supported by AMASS downlaoder.")

    filename = _DATASET_PATHS[dataset_name]['sfile']

    params = {
        'domain': 'amass',
        'sfile': filename
    }

    if not get_has_agreed_to_license(dataset_name, amass_session_id):
        # ask the user to agree to the license
        license_text = get_license_text(_DATASET_PATHS[dataset_name]['licensename'])
        prompt_msg = r"""
          By using this script to download {filename}, you agree to the following license:
          {license_text}
          """.format(
            filename=filename,
            license_text=license_text
        )
        if click.confirm(prompt_msg, default=True):
            # send to the server that the user has accepted
            set_has_agreed_to_license(dataset_name, amass_session_id)
        else:
            # user did not agree to this license, do not download it.
            print("User refused to agree to the license.")
            return None

    return download_and_save_file(amass_session_id, url, params, headers, save_to, rename_to)


def download_human_ml_3d(amass_session_id: str, save_to: Path) -> None:
    for dataset in HUMAN_ML3D_DATASET_NAMES:
        dataset_name, folder_save_as_name = dataset
        download_amass_file_smpl_h_g(
            amass_session_id, dataset_name, save_to,
            rename_to=Path(folder_save_as_name + ".tar.bz2")
        )
