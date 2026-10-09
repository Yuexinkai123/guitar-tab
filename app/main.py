from __future__ import annotations

import logging
import json
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import soundfile as sf

from .transcribe import analyze
from .score_export import export_midi, export_musicxml

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
POOL = ThreadPoolExecutor(max_workers=1)
LIMIT = 40 * 1024 * 1024
app = FastAPI(title="弦外 · 音乐转六线谱")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


def save_job(job_id):
    """Keep completed work available when the local server is updated/restarted."""
    try:
        with LOCK:
            snapshot = dict(JOBS[job_id])
        folder = DATA / job_id
        temporary = folder / 'job.json.tmp'
        temporary.write_text(json.dumps(snapshot, ensure_ascii=False), encoding='utf-8')
        temporary.replace(folder / 'job.json')
    except (OSError, KeyError):
        logging.exception('Could not save job: %s', job_id)


def restore_jobs():
    recovered = []
    for path in DATA.glob('*/job.json'):
        try:
            if len(path.parent.name) != 32 or any(c not in '0123456789abcdef' for c in path.parent.name):
                continue
            job = json.loads(path.read_text(encoding='utf-8'))
            if (job['id'] == path.parent.name and job['status'] in ('done', 'error')
                    and isinstance(job['created'], (float, int))):
                recovered.append(job)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    JOBS.update({job['id']: job for job in sorted(recovered, key=lambda job: job['created'])[-12:]})


restore_jobs()


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.post('/api/export/{format}')
def export_score(format: str, score: dict):
    if format not in ('midi', 'musicxml'):
        raise HTTPException(400, '不支持的乐谱格式。')
    try:
        content = export_midi(score) if format == 'midi' else export_musicxml(score)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(400, str(exc)) from exc
    return Response(content, media_type='audio/midi' if format == 'midi' else 'application/vnd.recordare.musicxml+xml')


def run_job(job_id, path, start, seconds, mode):
    def report(progress, message):
        with LOCK:
            JOBS[job_id].update(status="running", progress=progress, message=message)
    try:
        result = analyze(path, start, seconds, mode, DATA / "models", path.parent / "melody.wav", report)
        with LOCK:
            JOBS[job_id].update(status="done", progress=100, message="六线谱已生成", result=result)
    except ValueError as exc:
        with LOCK:
            JOBS[job_id].update(status="error", message=str(exc))
    except sf.LibsndfileError:
        with LOCK:
            JOBS[job_id].update(status="error", message="无法读取这个音频文件，请尝试重新导出为 MP3 或 WAV。")
    except Exception:
        logging.exception("Transcription failed: %s", job_id)
        with LOCK:
            JOBS[job_id].update(status="error", message="处理失败。首次运行请检查模型下载是否成功；详细原因可在程序窗口查看。")
    finally:
        save_job(job_id)


@app.post("/api/transcribe", status_code=202)
async def transcribe(file: UploadFile = File(...), start: float = Form(0),
                     seconds: float = Form(0), mode: str = Form("song")):
    if (not 0 <= start <= 7200 or not (seconds == 0 or 5 <= seconds <= 90)
            or (seconds == 0 and start != 0) or mode not in ("song", "solo", "guitar", "guitar_solo")):
        raise HTTPException(400, "请选择 5～90 秒片段，或从开头分析整首歌曲。")
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp3", ".wav", ".flac", ".ogg"}:
        raise HTTPException(400, "目前支持 MP3、WAV、FLAC 和 OGG 文件。")
    with LOCK:
        expired = [key for key, job in JOBS.items() if time.time() - job["created"] > 3600
                   and job["status"] in ("done", "error")]
        for key in expired:
            shutil.rmtree(DATA / key, ignore_errors=True)
            del JOBS[key]
        if sum(job["status"] in ("queued", "running") for job in JOBS.values()) >= 2:
            raise HTTPException(429, "已有任务正在处理，请等完成后再试。")
        if len(JOBS) >= 12:
            raise HTTPException(429, "本次运行已保留 12 份结果，请重启程序后继续。")
        job_id = uuid.uuid4().hex
        JOBS[job_id] = {"id": job_id, "status": "queued", "progress": 0,
                        "message": "等待分析", "created": time.time()}
    folder = DATA / job_id
    folder.mkdir()
    path = folder / ("original" + suffix)
    size = 0
    try:
        with path.open("wb") as target:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > LIMIT:
                    raise HTTPException(413, "文件超过 40 MB，请压缩或截取后上传。")
                target.write(chunk)
        if size == 0:
            raise HTTPException(400, "文件是空的，请重新选择。")
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        with LOCK:
            JOBS.pop(job_id, None)
        raise
    finally:
        await file.close()
    POOL.submit(run_job, job_id, path, start, seconds, mode)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if job is None:
            raise HTTPException(404, "任务不存在或已过期，请重新上传。")
        return dict(job)


@app.get("/api/jobs/{job_id}/melody")
def melody(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if job is None or job["status"] != "done":
            raise HTTPException(404, "音频尚未生成。")
    preview = DATA / job_id / 'preview.wav'
    return FileResponse(preview if preview.is_file() else DATA / job_id / "melody.wav", media_type="audio/wav")


@app.get('/api/jobs/{job_id}/original')
def original_audio(job_id: str):
    with LOCK:
        job = JOBS.get(job_id)
        if job is None or job['status'] != 'done':
            raise HTTPException(404, '原录音不存在或任务尚未完成。')
    for suffix in ('.mp3', '.wav', '.flac', '.ogg'):
        path = DATA / job_id / ('original' + suffix)
        if path.is_file():
            return FileResponse(path)
    raise HTTPException(404, '原录音不存在。')
