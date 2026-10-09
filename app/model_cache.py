"""Download the fixed vendor checkpoint in resumable, verified chunks."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import threading
import time

import requests

MODEL_URL = 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th'
MODEL_FILE = '955717e8-8726e21a.th'
MODEL_SIZE = 84141911
HASH_PREFIX = '8726e21a'
MODELS = {
    'htdemucs': (MODEL_FILE, MODEL_SIZE, HASH_PREFIX),
    'htdemucs_6s': ('5c90dfd2-34c22ccb.th', 54996327, '34c22ccb'),
}


def ensure_checkpoint(model_dir: Path, report, model_name='htdemucs'):
    model_file, model_size, hash_prefix = MODELS[model_name]
    model_url = 'https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/' + model_file
    folder = model_dir / 'checkpoints'
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / model_file
    if target.is_file():
        if hashlib.sha256(target.read_bytes()).hexdigest().startswith(hash_prefix):
            return target
        target.unlink()
    chunk_size = 4 * 1024 * 1024
    count = (model_size + chunk_size - 1) // chunk_size
    complete = 0
    lock = threading.Lock()

    def fetch(index):
        nonlocal complete
        start, end = index * chunk_size, min((index + 1) * chunk_size, model_size) - 1
        part = folder / f'{model_file}.{index}.part'
        if not part.is_file() or part.stat().st_size != end - start + 1:
            for attempt in range(4):
                try:
                    response = requests.get(model_url + f'?chunk={index}',
                                            headers={'Range': f'bytes={start}-{end}'}, timeout=60)
                    response.raise_for_status()
                    if (response.headers.get('Content-Range') != f'bytes {start}-{end}/{model_size}'
                            or len(response.content) != end - start + 1):
                        raise ValueError('Incomplete model download')
                    part.write_bytes(response.content)
                    break
                except (requests.RequestException, ValueError):
                    if attempt == 3:
                        raise RuntimeError('模型下载失败，请检查网络后再试。已完成的下载片段会保留。')
                    time.sleep(1)
        with lock:
            complete += 1
            report(12 + round(complete / count * 10), f'下载声部分离模型：{complete}/{count}（仅首次需要）')
        return part

    with ThreadPoolExecutor(max_workers=4) as pool:
        parts = list(pool.map(fetch, range(count)))
    digest = hashlib.sha256()
    temporary = target.with_suffix('.download')
    with temporary.open('wb') as output:
        for part in parts:
            contents = part.read_bytes()
            digest.update(contents)
            output.write(contents)
    if not digest.hexdigest().startswith(hash_prefix):
        temporary.unlink(missing_ok=True)
        for part in parts:
            part.unlink(missing_ok=True)
        raise RuntimeError('模型下载校验失败，请重试。')
    temporary.replace(target)
    for part in parts:
        part.unlink(missing_ok=True)
    return target
