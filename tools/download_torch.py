"""Recover interrupted wheel downloads using HTTP ranges and the vendor's hash."""
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import requests

URL = 'https://download-r2.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp312-cp312-win_amd64.whl'
SIZE = 206493444
EXPECTED = '4027d982eb2781c93825ab9527f17fbbb12dbabf422298e4b954be60016f87d8'
DEST = Path(__file__).resolve().parent.parent / 'data' / 'install'
DEST.mkdir(parents=True, exist_ok=True)
CHUNK = 4 * 1024 * 1024


def fetch(index):
    start, end = index * CHUNK, min((index + 1) * CHUNK, SIZE) - 1
    path = DEST / f'{index}.part'
    if path.exists() and path.stat().st_size == end - start + 1:
        return index
    for attempt in range(5):
        try:
            response = requests.get(URL + f'?chunk={index}', headers={'Range': f'bytes={start}-{end}'}, timeout=60)
            response.raise_for_status()
            if response.headers.get('Content-Range') != f'bytes {start}-{end}/{SIZE}' or len(response.content) != end - start + 1:
                raise ValueError('Incomplete or unexpected download range')
            path.write_bytes(response.content)
            print(f'Part {index + 1}/{(SIZE + CHUNK - 1) // CHUNK}', flush=True)
            return index
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1)


if __name__ == '__main__':
    count = (SIZE + CHUNK - 1) // CHUNK
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(fetch, range(count)))
    wheel = DEST / 'torch-2.6.0+cpu-cp312-cp312-win_amd64.whl'
    digest = hashlib.sha256()
    with wheel.open('wb') as output:
        for index in range(count):
            chunk = (DEST / f'{index}.part').read_bytes()
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != EXPECTED:
        raise RuntimeError('Vendor SHA256 verification failed; do not install this wheel.')
    print('Verified wheel:', wheel, flush=True)
