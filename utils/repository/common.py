import requests
from typing import Dict, Optional
from tqdm.auto import tqdm
from pathlib import Path

USER_AGENT = 'Mozilla/5.0 (Linux; Linux x86_64; en-US) Gecko/20100101 Firefox/71.7'


def get_login_headers(host: str) -> Dict[str, str]:
    return {
        'Host': host,  # e.g. 'smpl.is.tue.mpg.de'
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
        'sec-ch-ua-mobile': '?0',
        'Upgrade-Insecure-Requests': '1',
        'Origin': f'https://{host}',
        'User-Agent': USER_AGENT,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,'
                  'image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Sec-Fetch-Site': 'same-origin',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-User': '?1',
        'Sec-Fetch-Dest': 'document',
        'Referer': f'https://{host}login.php',
        'Accept-Language': 'en-US,en;q=0.9',
        'Content-Type': 'application/x-www-form-urlencoded',
    }


def get_login_session_id(host: str, username: str, password: str) -> str:
    headers = get_login_headers(host)
    data = {
        'username': username,
        'password': password,
        'commit': 'Log in',
    }
    session = requests.Session()
    session.post(f'https://{host}/login.php', headers=headers, data=data)
    cookies = session.cookies.get_dict()
    return cookies["PHPSESSID"]


def download_and_save_file(session_id: str, url: str, params: Dict[str, str], headers: Dict[str, str],
                           save_to: Path, rename_to: Optional[Path] = None) -> Path:
    cookies = {
        'PHPSESSID': session_id,
    }

    to_download = params["sfile"]

    response = requests.get(
        url,
        params=params,
        cookies=cookies,
        headers=headers,
        stream=True
    )

    # the download param may be a full filepath, find the name of the file from it
    local_filename = Path(to_download).parts[-1]
    p = save_to / (rename_to or local_filename)

    # credit to: https://stackoverflow.com/a/61575758/23160579
    with tqdm.wrapattr(open(str(p), "wb"), "write", miniters=1,
                       total=int(response.headers.get('content-length', 0)),
                       desc=f"Downloading {to_download}") as f:
        for chunk in response.iter_content(chunk_size=4096):
            f.write(chunk)
    return p
