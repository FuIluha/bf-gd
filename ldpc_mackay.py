"""
LDPC codes from D. MacKay's Encyclopedia of Sparse Graph Codes
(http://www.inference.org.uk/mackay/codes/data.html)
"""
import gzip
import json
import os
import argparse
import urllib.request

from ldpc_common.alist import Alist

CODES_DIR = 'codes'
BASE_URL = 'http://www.inference.org.uk/mackay/codes/EN/C/'

# Code name -> True if the alist file is stored gzipped
CODES = {
    '96.33.964': False,
    '204.33.484': False,
    'PEGReg252x504': True,
    'PEGReg504x1008': True,
}


def download_alist(name):
    """
    Download the parity check matrix in the ALIST format
    """
    url = BASE_URL + name + ('.gz' if CODES[name] else '')
    with urllib.request.urlopen(url) as response:
        data = response.read()
    return gzip.decompress(data) if CODES[name] else data


def get_mackay_code(name):
    """
    Main function
    """
    filename_template = f'mackay_{name}'
    pcm_file = filename_template + '.alist'
    pcm_path = os.path.join(CODES_DIR, pcm_file)

    print(f'Downloading {name}...')
    with open(pcm_path, 'wb') as filedesc:
        filedesc.write(download_alist(name))
    n_checks, block_len = Alist.read(pcm_path).shape
    print(f'Parity check matrix: {n_checks} x {block_len}')

    code = {
        'name': filename_template,
        'pcm': pcm_file,
        'punctured': 0,
        'is_systematic': False
    }
    json_file = os.path.join(CODES_DIR, filename_template + '.json')
    with open(json_file, 'w', encoding='utf-8') as filedesc:
        json.dump(code, filedesc, indent=2)
    print(f'Successfully generated {json_file}')
    return json_file


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download LDPC codes from MacKay\'s encyclopedia')
    parser.add_argument('--code', choices=list(CODES) + ['all'], required=True,
                        help='Code name or \'all\'')
    args = parser.parse_args()
    for code_name in (CODES if args.code == 'all' else [args.code]):
        get_mackay_code(code_name)
