# server.py
import os
import sys
import time
import glob
import io
import threading
import queue
import subprocess
from typing import Optional, Tuple

from collections import deque
_LOGS = deque(maxlen=200)

from fastapi import FastAPI, Request, Header
from fastapi.responses import (
    HTMLResponse,
    StreamingResponse,
    JSONResponse,
    PlainTextResponse,
)

APP = FastAPI()

# ---------------------------
# Configuration (env vars)
# ---------------------------
MG2_IMG_PATH   = os.environ.get("MG2_IMG_PATH", "/workspace/Matrix/Matrix-Game-2/images/image.png")
MG2_OUTPUT_DIR = os.environ.get("MG2_OUTPUT_DIR", "/workspace/Matrix/Matrix-Game-2/outputs/universal_run")
MG2_REPO_DIR   = os.environ.get("MG2_REPO_DIR", "/workspace/Matrix/Matrix-Game-2")
MG2_CONFIG     = os.environ.get("MG2_CONFIG", "configs/inference_yaml/inference_universal.yaml")
MG2_CKPT       = os.environ.get("MG2_CKPT", "Matrix-Game-2.0/base_distilled_model/base_distill.safetensors")
MG2_PRE        = os.environ.get("MG2_PRE", "Matrix-Game-2.0")
MG2_SEED       = os.environ.get("MG2_SEED", "42")

os.makedirs(MG2_OUTPUT_DIR, exist_ok=True)

# ---------------------------
# Launch Matrix-Game-2.0
# ---------------------------
cmd = [
    sys.executable, "-u", "inference_streaming.py",   # -u for unbuffered output
    "--config_path", MG2_CONFIG,
    "--checkpoint_path", MG2_CKPT,
    "--output_folder", MG2_OUTPUT_DIR,
    "--seed", MG2_SEED,
    "--pretrained_model_path", MG2_PRE,
]

proc = subprocess.Popen(
    cmd,
    cwd=MG2_REPO_DIR,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
    universal_newlines=True,
)

# ---------------------------
# CLI prompt synchronization
# ---------------------------
_in_q: "queue.Queue[Tuple[str, str] | str]" = queue.Queue()
_ready_mouse = False
_ready_move = False
_started = False
_lock = threading.Lock()

def _reader():
    """Read MG2 stdout, detect prompts, answer image path once."""
    global _ready_mouse, _ready_move, _started
    for line in proc.stdout:
        s = line.rstrip()
        print("[MG2]", s, flush=True)
        _LOGS.append(s)

        if "Please input the image path" in s and not _started:
            if proc.stdin:
                proc.stdin.write(MG2_IMG_PATH + "\n")
                proc.stdin.flush()
            _started = True

        elif "Please input the mouse action" in s:
            with _lock:
                _ready_mouse = True
                _ready_move = False

        # Some versions explicitly ask for movement; others imply it after mouse.
        elif "movement action" in s or "PRESS [W, S, A, D, Q]" in s:
            with _lock:
                _ready_move = True

def _writer():
    """Send (mouse, move) pairs to the process in sync with prompts."""
    global _ready_mouse, _ready_move
    while True:
        item = _in_q.get()
        if item == "STOP":
            break
        mouse, move = item

        # wait for mouse prompt
        while True:
            with _lock:
                if _ready_mouse:
                    break
            time.sleep(0.01)

        if proc.stdin:
            proc.stdin.write((mouse or "U") + "\n")
            proc.stdin.flush()

        with _lock:
            _ready_mouse = False
            _ready_move = True

        # wait for move prompt
        while True:
            with _lock:
                if _ready_move:
                    break
            time.sleep(0.01)

        if proc.stdin:
            proc.stdin.write((move or "Q") + "\n")
            proc.stdin.flush()

        with _lock:
            _ready_move = False

threading.Thread(target=_reader, daemon=True).start()
threading.Thread(target=_writer, daemon=True).start()

# ---------------------------
# Find/serve the latest mp4
# ---------------------------
def _latest_mp4() -> Optional[str]:
    # Prefer *_current.mp4; fall back to any .mp4 under output dir
    mp4s = glob.glob(os.path.join(MG2_OUTPUT_DIR, "**", "*_current.mp4"), recursive=True)
    if not mp4s:
        mp4s = glob.glob(os.path.join(MG2_OUTPUT_DIR, "**", "*.mp4"), recursive=True)
    if not mp4s:
        return None
    mp4s.sort(key=lambda p: os.path.getmtime(p))
    return mp4s[-1]

def _current_mp4_path() -> Optional[str]:
    return _latest_mp4()

@APP.get("/meta")
def meta():
    """Return mtime/size so the client knows when to reload the video."""
    p = _current_mp4_path()
    if not p or not os.path.exists(p):
        return {"exists": False}
    return {
        "exists": True,
        "path": os.path.basename(p),
        "mtime": os.path.getmtime(p),
        "size": os.path.getsize(p),
    }

@APP.get("/current.mp4")
def current_mp4(range: str | None = Header(default=None)):
    """
    Serve the entire latest *_current.mp4 with HTTP Range support,
    so <video> can stream/seek. When MG2 overwrites the file, the
    /meta mtime changes and the page reloads the src.
    """
    path = _current_mp4_path()
    if not path or not os.path.exists(path):
        # Empty stream placeholder (no file yet)
        return StreamingResponse(iter(()), media_type="video/mp4",
                                 headers={"Cache-Control": "no-store"})

    file_size = os.path.getsize(path)
    start = 0
    end = file_size - 1

    if range and range.startswith("bytes="):
        try:
            parts = range.replace("bytes=", "").split("-")
            start = int(parts[0]) if parts[0] else 0
            end = int(parts[1]) if len(parts) > 1 and parts[1] else end
        except Exception:
            start, end = 0, file_size - 1
        start = max(0, start)
        end = min(end, file_size - 1)

    def _read():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            chunk = 1024 * 1024
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Type": "video/mp4",
        "Accept-Ranges": "bytes",
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Content-Length": str(end - start + 1),
        "Content-Range": f"bytes {start}-{end}/{file_size}",
    }
    status = 206 if range else 200
    return StreamingResponse(_read(), status_code=status, headers=headers)

# ---------------------------
# Control UI + endpoints
# ---------------------------
INDEX_HTML = """
<!doctype html>
<title>Matrix-Game 2.0 — Live Control</title>
<style>
  body{font-family:system-ui,sans-serif;margin:1rem}
  .row{display:flex;gap:.5rem;margin:.5rem 0;flex-wrap:wrap}
  button{padding:.5rem .75rem;font-size:1rem}
  video{max-width:100%;border:1px solid #ccc;border-radius:8px}
  label{margin-right:.25rem}
  select{padding:.25rem .5rem}
  .muted{color:#666;font-size:.9rem}
</style>
<h1>Matrix-Game 2.0 — Live Control</h1>
<div class="row muted"><strong>Keys:</strong> Mouse: I K J L U &nbsp;|&nbsp; Move: W A S D Q</div>
<div class="row">
  <label>Mouse:</label>
  <select id="mouse"><option>U</option><option>I</option><option>K</option><option>J</option><option>L</option></select>
  <label>Move:</label>
  <select id="move"><option>Q</option><option>W</option><option>A</option><option>S</option><option>D</option></select>
  <button onclick="sendCmd()">Send</button>
</div>
<div class="row">
  <button onclick="quick('U','W')">U+W</button>
  <button onclick="quick('U','S')">U+S</button>
  <button onclick="quick('U','A')">U+A</button>
  <button onclick="quick('U','D')">U+D</button>
  <button onclick="quick('I','Q')">I+Q</button>
  <button onclick="quick('K','Q')">K+Q</button>
  <button onclick="quick('J','Q')">J+Q</button>
  <button onclick="quick('L','Q')">L+Q</button>
</div>

<video id="vid" controls autoplay playsinline muted></video>
<div class="row muted" id="status"></div>

<script>
const vid = document.getElementById('vid');
const statusEl = document.getElementById('status');

async function sendCmd(){
  const mouse=document.getElementById('mouse').value;
  const move=document.getElementById('move').value;
  await fetch('/cmd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mouse,move})});
}

async function quick(m,v){
  await fetch('/cmd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mouse:m,move:v})});
}

// Keyboard shortcuts
document.addEventListener('keydown', async (e)=>{
  const k=e.key.toLowerCase();
  const move={'w':'W','a':'A','s':'S','d':'D','q':'Q'}[k];
  const cam ={'i':'I','k':'K','j':'J','l':'L','u':'U'}[k];
  if(move){ await quick('U',move); }
  if(cam){  await quick(cam,'Q'); }
});

// Poll for new 1s clip; reload <video> when mtime/size changes
let lastTag = null;
async function refreshIfChanged(){
  try{
    const r = await fetch('/meta');
    const j = await r.json();
    if(!j.exists){ statusEl.textContent='Waiting for first clip…'; return; }
    const tag = `${j.mtime}-${j.size}`;
    statusEl.textContent = `Current: ${j.path}  |  size: ${j.size}  |  mtime: ${new Date(j.mtime*1000).toLocaleTimeString()}`;
    if(tag !== lastTag){
      lastTag = tag;
      const url = `/current.mp4?t=${encodeURIComponent(tag)}`;
      const wasPaused = vid.paused;
      vid.src = url;                       // reload full clip
      await vid.play().catch(()=>{});
      if (wasPaused) vid.pause();
    }
  }catch(e){
    // ignore transient errors
  }
}
setInterval(refreshIfChanged, 250);
refreshIfChanged();
</script>
"""

@APP.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML

@APP.post("/cmd")
async def cmd_endpoint(req: Request):
    data = await req.json()
    mouse = (data.get("mouse") or "U").upper()[0]
    move  = (data.get("move")  or "Q").upper()[0]
    # Enqueue the pair; writer thread syncs with prompts
    _in_q.put((mouse, move))
    return JSONResponse({"ok": True, "mouse": mouse, "move": move})

@APP.get("/favicon.ico")
def favicon():
    return PlainTextResponse("", status_code=204)

@APP.get("/healthz")
def health():
    alive = (proc.poll() is None)
    cur = _current_mp4_path()
    return {"proc_alive": alive, "latest_mp4": cur, "output_dir": MG2_OUTPUT_DIR}

@APP.get("/logs")
def logs():
    return JSONResponse({"lines": list(_LOGS)})