from pathlib import Path

from utils.repository.common import get_login_session_id, download_and_save_file, USER_AGENT


def get_smpl_session_id(username: str, password: str) -> str:
    host = "smpl.is.tue.mpg.de"
    return get_login_session_id(host, username, password)


def download_smpl_m_f_models(smpl_session_id: str, save_to: Path) -> Path:
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
        'Referer': 'https://smpl.is.tue.mpg.de/',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    params = {
        'domain': 'smpl',
        'sfile': 'SMPL_python_v.1.1.0.zip',
    }

    return download_and_save_file(smpl_session_id, url, params, headers, save_to)


def download_dmpls(smpl_session_id: str, save_to: Path) -> Path:
    url = 'https://download.is.tue.mpg.de/download.php'
    headers = {
        'Host': 'download.is.tue.mpg.de',
        'Connection': 'keep-alive',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'sec-ch-ua-mobile': '?0',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,'
                  'image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Sec-Fetch-Site': 'same-site',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-User': '?1',
        'Sec-Fetch-Dest': 'document',
        'Referer': 'https://smpl.is.tue.mpg.de/',
        'Accept-Language': 'en-US,en;q=0.9',
    }
    params = {
        'domain': 'smpl',
        'sfile': 'dmpls.tar.xz',
    }
    return download_and_save_file(smpl_session_id, url, params, headers, save_to)
