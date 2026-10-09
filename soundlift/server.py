"""Local web app: paste a YouTube link, get a playlist of its music + sound effects."""

from __future__ import annotations

import io
import json
import queue
import threading
import traceback
import uuid
import zipfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import pipeline

WEB = Path(__file__).parent / "web"

# Share of the progress bar each stage gets (identify is slow: Shazam is rate limited).
STAGES = {"download": 0.05, "separate": 0.10, "tag": 0.05, "identify": 0.25, "sfx": 0.10, "clean": 0.40, "export": 0.05}


class JobIn(BaseModel):
    url: str


def create_app(out_root: Path) -> FastAPI:
    out_root.mkdir(parents=True, exist_ok=True)
    app = FastAPI(title="soundlift")
    jobs: dict[str, dict] = {}
    q: queue.Queue[str] = queue.Queue()

    def worker() -> None:
        # One job at a time: the models are heavy and Shazam doesn't like bursts.
        while True:
            jid = q.get()
            job = jobs[jid]
            job["status"] = "running"

            def progress(stage: str, frac: float) -> None:
                order = list(STAGES)
                done = sum(STAGES[s] for s in order[: order.index(stage)])
                job["stage"] = stage
                job["progress"] = round(done + STAGES[stage] * frac, 3)

            try:
                res = pipeline.run(job["url"], out_root, progress)
                job.update(status="done", progress=1.0, video_id=res.manifest["id"])
            except Exception as e:  # surface the error to the UI instead of dying
                traceback.print_exc()
                job.update(status="error", error=str(e) or e.__class__.__name__)

    threading.Thread(target=worker, daemon=True).start()

    @app.post("/api/jobs")
    def create_job(body: JobIn):
        url = body.url.strip()
        if not url:
            raise HTTPException(400, "Paste a YouTube link")
        jid = uuid.uuid4().hex[:12]
        jobs[jid] = {"id": jid, "url": url, "status": "queued", "stage": None, "progress": 0.0,
                     "position": q.qsize()}
        q.put(jid)
        return jobs[jid]

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        if jid not in jobs:
            raise HTTPException(404)
        return jobs[jid]

    @app.get("/api/videos")
    def list_videos():
        out = []
        for m in sorted(out_root.glob("*/manifest.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            d = json.loads(m.read_text())
            out.append({k: d.get(k) for k in ("id", "title", "channel", "thumbnail")} | {"count": len(d["items"])})
        return out

    @app.get("/api/videos/{vid}")
    def get_video(vid: str):
        m = out_root / vid / "manifest.json"
        if not m.exists():
            raise HTTPException(404)
        return json.loads(m.read_text())

    @app.get("/api/videos/{vid}/zip")
    def zip_video(vid: str):
        d = out_root / vid
        if not (d / "manifest.json").exists():
            raise HTTPException(404)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(d / "manifest.json", "manifest.json")
            for f in sorted((d / "clips").glob("*.wav")):
                z.write(f, f"clips/{f.name}")
        buf.seek(0)
        headers = {"Content-Disposition": f'attachment; filename="soundlift-{vid}.zip"'}
        return StreamingResponse(buf, media_type="application/zip", headers=headers)

    app.mount("/files", StaticFiles(directory=out_root), name="files")

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    return app


def serve(out_root: Path, port: int = 8765) -> None:
    import uvicorn

    print(f"soundlift running at http://localhost:{port}")
    uvicorn.run(create_app(out_root), host="127.0.0.1", port=port, log_level="warning")
