import click
import getpass
from pathlib import Path
from utils.repository.smplx import get_smplx_session_id, download_vposer_model
from utils.repository.smpl import get_smpl_session_id, download_smpl_m_f_models, download_dmpls
from utils.repository.smplh import get_smplh_session_id, download_smplh_models
from utils.repository.amass import get_amass_session_id, download_human_ml_3d

_REPO_ROOT = Path(__file__).parent.parent.parent

if __name__ == "__main__":
    # download HumanML3D
    prompt_msg = """
    By using this script to download the AMASS data for HumanML3D, you agree to their terms and conditions. 
    The license information may be read here: https://amass.is.tue.mpg.de/license.html

    Do you agree to their terms of use?

    If you agree, you will be prompted for login credentials (username / password) that you 
    created for their website. If you require an account, please make one here: 

    https://amass.is.tue.mpg.de/register.php
    
    Each individual dataset may also have its own license, which will be shown in this prompt when
    applicable. The dataset will not download unless you agree to those terms as well. 
    """
    if click.confirm(prompt_msg, default=True):
        print(
            "Please enter the login credentials you have created for "
            "AMASS: https://amass.is.tue.mpg.de/index.html"
        )
        username = input("Username: ")
        password = getpass.getpass("Password: ")
        try:
            sess_id = get_amass_session_id(username, password)
        except Exception as e:
            print(f"Unable to obtain AMASS session.")
            raise e

        # save to the root of the repo
        download_human_ml_3d(amass_session_id=sess_id, save_to=_REPO_ROOT)
    else:
        print("Cancelled AMASS/HumanML3D download.")

    # get Vposer from SMPLX
    prompt_msg = """
    By using this script to download the SMPLX data, you agree to their terms and conditions. 
    The license information may be read here: https://smpl-x.is.tue.mpg.de/modellicense.html
    
    Do you agree to their terms of use?
    
    If you agree, you will be prompted for login credentials (username / password) that you 
    created for their website. If you require an account, please make one here: 
    
    https://smpl-x.is.tue.mpg.de/register.php
    """
    if click.confirm(prompt_msg, default=True):
        print(
            "Please enter the login credentials you have created for "
            "SMPLX: https://smpl-x.is.tue.mpg.de/index.html"
        )
        username = input("Username: ")
        password = getpass.getpass("Password: ")
        try:
            sess_id = get_smplx_session_id(username, password)
        except Exception as e:
            print(f"Unable to obtain SMPLX session.")
            raise e

        # save to the root of the repo
        p = download_vposer_model(sess_id, save_to=_REPO_ROOT)
        print(f"Successfully saved vposer model weights to: {p}")
    else:
        print("Cancelled VPoser model download.")

    # download the male and female SMPL models
    prompt_msg = """
    By using this script to download the SMPL data, you agree to their terms and conditions. 
    The license information may be read here: https://smpl.is.tue.mpg.de/modellicense.html

    Do you agree to their terms of use?

    If you agree, you will be prompted for login credentials (username / password) that you 
    created for their website. If you require an account, please make one here: 

    https://smpl.is.tue.mpg.de/register.php
    """
    if click.confirm(prompt_msg, default=True):
        print(
            "Please enter the login credentials you have created for "
            "SMPL: https://smpl.is.tue.mpg.de/index.html"
        )
        username = input("Username: ")
        password = getpass.getpass("Password: ")
        try:
            sess_id = get_smpl_session_id(username, password)
        except Exception as e:
            print(f"Unable to obtain SMPL session.")
            raise e

        # save to the root of the repo
        p = download_smpl_m_f_models(sess_id, save_to=_REPO_ROOT)
        print(f"Successfully saved SMPL python library to: {p}")

        p = download_dmpls(sess_id, save_to=_REPO_ROOT)
        print(f"Successfully saved DMPLS to: {p}")
    else:
        print("Cancelled SMPL model download.")

    # download the male, female, and neutral SMPLH models
    prompt_msg = """
       By using this script to download the SMPLH data, you agree to their terms and conditions. 
       The license information may be read here: https://mano.is.tue.mpg.de/license.html

       Do you agree to their terms of use?

       If you agree, you will be prompted for login credentials (username / password) that you 
       created for their website. If you require an account, please make one here: 

       https://mano.is.tue.mpg.de/register.php
       """
    if click.confirm(prompt_msg, default=True):
        print(
            "Please enter the login credentials you have created for SMPLH / Mano: "
            "https://mano.is.tue.mpg.de/index.html"
        )
        username = input("Username: ")
        password = getpass.getpass("Password: ")
        try:
            sess_id = get_smplh_session_id(username, password)
        except Exception as e:
            print(f"Unable to obtain SMPLH / Mano session.")
            raise e

        # save to the root of the repo
        p = download_smplh_models(sess_id, save_to=_REPO_ROOT)
        print(f"Successfully saved SMPLH python library to: {p}")
    else:
        print("Cancelled SMPLH model download.")
