# /workspace/mg2_server.py
import os, sys, time, glob, threading, queue, subprocess, io
from typing import Optional
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from PIL import Image

"""
MG2 Web Control + Live Stream Server
------------------------------------
- Spawns Matrix-Game-2.0 interactive process
- Feeds it inputs coming from the browser
- Streams latest frame(s) from outputs folder as MJPEG

ENV VARS (set before launching uvicorn):
  MG2_IMG_PATH:    input image path (default: /workspace/Matrix/Matrix-Game-2/images/image.png)
  MG2_OUTPUT_DIR:  output frames dir (default: /workspace/Matrix/Matrix-Game-2/outputs/universal_run)
  MG2_REPO_DIR:    repo dir (default: /workspace/Matrix/Matrix-Game-2)
  MG2_CONFIG:      config path (default: configs/inference_yaml/inference_universal.yaml)
  MG2_CKPT:        checkpoint path (default: Matrix-Game-2.0/base_distilled_model/base_distill.safetensors)
  MG2_PRE:         pretrained dir (default: Matrix-Game-2.0)
"""

APP = FastAPI()

MG2_IMG_PATH   = os.environ.get("MG2_IMG_PATH", "/workspace/Matrix/Matrix-Game-2/images/image.png")
MG2_OUTPUT_DIR = os.environ.get("MG2_OUTPUT_DIR", "/workspace/Matrix/Matrix-Game-2/outputs/universal_run")
MG2_REPO_DIR   = os.environ.get("MG2_REPO_DIR", "/workspace/Matrix/Matrix-Game-2")
MG2_CONFIG     = os.environ.get("MG2_CONFIG", "configs/inference_yaml/inference_universal.yaml")
MG2_CKPT       = os.environ.get("MG2_CKPT", "Matrix-Game-2.0/base_distilled_model/base_distill.safetensors")
MG2_PRE        = os.environ.get("MG2_PRE", "Matrix-Game-2.0")

os.makedirs(MG2_OUTPUT_DIR, exist_ok=True)

# --- subprocess launch & I/O ---

cmd = [
    sys.executable, "inference_streaming.py",
    "--config_path", MG2_CONFIG,
    "--checkpoint_path", MG2_CKPT,
    "--output_folder", MG2_OUTPUT_DIR,
    "--seed", "42",
    "--pretrained_model_path", MG2_PRE,
]

# run from repo dir
proc = subprocess.Popen(
    cmd,
    cwd=MG2_REPO_DIR,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    bufsize=1,
    universal_newlines=True,
)

# Feed thread writes to proc.stdin; we push pairs (mouse, move)
_in_q: "queue.Queue[tuple[str,str]|str]" = queue.Queue()
_state_lock = threading.Lock()
_ready_for_mouse = False
_ready_for_move = False
_started = False

def _reader_thread():
    """Parse script prompts; auto-answer image path; mark states so we know which input to send."""
    global _ready_for_mouse, _ready_for_move, _started
    for line in proc.stdout:
        line_str = line.rstrip()
        # print to server logs for debugging
        print("[MG2]", line_str, flush=True)

        if "Please input the image path" in line_str and not _started:
            # send the image path
            if proc.stdin:
                proc.stdin.write(MG2_IMG_PATH + "\n")
                proc.stdin.flush()
            _started = True
        elif "Please input the mouse action" in line_str:
            with _state_lock:
                _ready_for_mouse = True
                _ready_for_move = False
        elif "Please input the movement action" in line_str or "PRESS [W, S, A, D, Q]" in line_str:
            # some versions prompt explicitly, others implicitly after mouse
            with _state_lock:
                # If mouse was just sent, now expecting move
                _ready_for_move = True

reader_t = threading.Thread(target=_reader_thread, daemon=True)
reader_t.start()

def _writer_thread():
    """Consume queued commands and write them as two lines: <mouse>\n<move>\n"""
    while True:
        item = _in_q.get()
        if item == "STOP":
            break
        mouse, move = item  # tuple[str,str]
        # Wait until the script is ready for mouse, then move
        while True:
            with _state_lock:
                ready_mouse = _ready_for_mouse
            if ready_mouse: break
            time.sleep(0.02)

        if proc.stdin:
            proc.stdin.write((mouse or "U") + "\n")
            proc.stdin.flush()
        with _state_lock:
            # mouse consumed; now we expect move
            _ready_for_mouse = False
            _ready_for_move = True

        while True:
            with _state_lock:
                ready_move = _ready_for_move
            if ready_move: break
            time.sleep(0.02)

        if proc.stdin:
            proc.stdin.write((move or "Q") + "\n")
            proc.stdin.flush()
        with _state_lock:
            _ready_for_move = False  # consumed; next cycle script will ask again

writer_t = threading.Thread(target=_writer_thread, daemon=True)
writer_t.start()

# --- helpers for frames ---

def _latest_frame_path() -> Optional[str]:
    # support png/jpg; MG2 usually writes PNGs
    files = []
    files.extend(glob.glob(os.path.join(MG2_OUTPUT_DIR, "*.png")))
    files.extend(glob.glob(os.path.join(MG2_OUTPUT_DIR, "*.jpg")))
    if not files:
        # might be nested (e.g., outputs/universal_run/step_xx/)
        nested = glob.glob(os.path.join(MG2_OUTPUT_DIR, "**", "*.png"), recursive=True)
        nested += glob.glob(os.path.join(MG2_OUTPUT_DIR, "**", "*.jpg"), recursive=True)
        files = nested
    if not files:
        return None
    files.sort()
    return files[-1]

def _mjpeg_generator():
    boundary = "frame"
    last_mod = 0
    while True:
        path = _latest_frame_path()
        if path and os.path.exists(path):
            mtime = os.path.getmtime(path)
            if mtime != last_mod:
                last_mod = mtime
                try:
                    img = Image.open(path).convert("RGB")
                    # optionally resize here to reduce bandwidth
                    bio = io.BytesIO()
                    img.save(bio, format="JPEG", quality=85, optimize=True)
                    frame = bio.getvalue()
                    yield (b"--" + boundary.encode() + b"\r\n" +
                           b"Content-Type: image/jpeg\r\n" +
                           f"Content-Length: {len(frame)}\r\n\r\n".encode() +
                           frame + b"\r\n")
                except Exception as e:
                    # If frame is mid-write, just wait a tick
                    time.sleep(0.03)
                    continue
        time.sleep(0.03)

# --- routes ---

@APP.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<title>Matrix-Game 2.0 — Live</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 1rem; }
  .row { display:flex; gap:.5rem; margin:.5rem 0; flex-wrap: wrap; }
  button { padding:.5rem .75rem; font-size:1rem; }
  img { max-width: 100%; border:1px solid #ccc; border-radius:8px; }
</style>
<h1>Matrix-Game 2.0 — Live Control</h1>
<div class="row">
  <strong>Keys:</strong> Mouse: I K J L U &nbsp; | &nbsp; Move: W A S D Q
</div>
<div class="row">
  <label>Mouse:</label>
  <select id="mouse">
    <option>U</option><option>I</option><option>K</option><option>J</option><option>L</option>
  </select>
  <label>Move:</label>
  <select id="move">
    <option>Q</option><option>W</option><option>A</option><option>S</option><option>D</option>
  </select>
  <button onclick="sendCmd()">Send</button>
</div>

<div class="row">
  <button onclick="quick('U','W')">U + W</button>
  <button onclick="quick('U','S')">U + S</button>
  <button onclick="quick('U','A')">U + A</button>
  <button onclick="quick('U','D')">U + D</button>
  <button onclick="quick('I','Q')">I + Q</button>
  <button onclick="quick('K','Q')">K + Q</button>
  <button onclick="quick('J','Q')">J + Q</button>
  <button onclick="quick('L','Q')">L + Q</button>
</div>

<img id="stream" src="/mjpg" alt="live stream"/>

<script>
async function sendCmd() {
  const mouse = document.getElementById('mouse').value;
  const move  = document.getElementById('move').value;
  await fetch('/cmd', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({mouse, move})
  });
}
async function quick(m, v) {
  await fetch('/cmd', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({mouse:m, move:v})
  });
}
// Keyboard shortcuts
document.addEventListener('keydown', async (e)=>{
  const k = e.key.toLowerCase();
  const moveKeys = {w:'W',a:'A',s:'S',d:'D',q:'Q'};
  const camKeys  = {i:'I',k:'K',j:'J',l:'L',u:'U'};
  if (moveKeys[k]) { await quick('U', moveKeys[k]); }
  if (camKeys[k])  { await quick(camKeys[k], 'Q'); }
});
</script>
"""

@APP.get("/mjpg")
def mjpg():
    return StreamingResponse(_mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@APP.post("/cmd")
async def cmd(req: Request):
    data = await req.json()
    mouse = (data.get("mouse") or "U").upper()[0]
    move  = (data.get("move") or "Q").upper()[0]
    # Enqueue pair; writer thread will synchronize with prompts
    _in_q.put((mouse, move))
    return JSONResponse({"ok": True, "mouse": mouse, "move": move})

@APP.get("/healthz")
def health():
    alive = (proc.poll() is None)
    return {"proc_alive": alive, "output_dir": MG2_OUTPUT_DIR}