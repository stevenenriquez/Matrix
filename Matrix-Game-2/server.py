# server.py
import os, sys, time, glob, io, threading, queue, subprocess, pathlib, shutil
from typing import Optional, Tuple, List
from collections import deque

from fastapi import FastAPI, Request, Header, Query, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, PlainTextResponse, Response
from PIL import Image

APP = FastAPI()

# ---------------------------
# Configuration (env vars)
# ---------------------------
ROOT_DIR       = os.environ.get("MG2_REPO_DIR", "/workspace/Matrix/Matrix-Game-2")
IMAGES_DIR     = os.environ.get("MG2_IMAGES_DIR", os.path.join(ROOT_DIR, "images"))
OUTPUT_DIR     = os.environ.get("MG2_OUTPUT_DIR", os.path.join(ROOT_DIR, "outputs/universal_run"))
CONFIG_PATH    = os.environ.get("MG2_CONFIG", "configs/inference_yaml/inference_universal.yaml")
CKPT_PATH      = os.environ.get("MG2_CKPT", "Matrix-Game-2.0/base_distilled_model/base_distill.safetensors")
PRETRAIN_DIR   = os.environ.get("MG2_PRE", "Matrix-Game-2.0")
SEED           = os.environ.get("MG2_SEED", "42")

# Create output dir if missing
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Sanity: absolute, normalized paths
ROOT_DIR   = str(pathlib.Path(ROOT_DIR).resolve())
IMAGES_DIR = str(pathlib.Path(IMAGES_DIR).resolve())
OUTPUT_DIR = str(pathlib.Path(OUTPUT_DIR).resolve())

# ---------------------------
# Subprocess & I/O sync state
# ---------------------------
proc: Optional[subprocess.Popen] = None
proc_lock = threading.Lock()

# queues / flags
_in_q: "queue.Queue[Tuple[str, str] | str]" = queue.Queue()
_ready_mouse = False
_ready_move  = False
_await_image = False
io_lock = threading.Lock()

# logs buffer
_LOGS = deque(maxlen=400)

# current image in use
_current_image_path = os.environ.get("MG2_IMG_PATH", os.path.join(IMAGES_DIR, "image.png"))

# heuristics to detect the image-path prompt from the subprocess
_IMG_PROMPT_TOKENS = [
    "please input the image path:",
    "input the image path",
    "image path:",
    "please input the image",
    "please input image",
    "enter image path",
    "path of the image",
    "select an image",          # some scripts print this wording
    "select image",             # variants
    "choose an image",
]


def _within(base: str, path: str) -> bool:
    """Return True if path is inside base (after resolving)."""
    base_p = pathlib.Path(base).resolve()
    path_p = pathlib.Path(path).resolve()
    try:
        return base_p == pathlib.Path(os.path.commonpath([base_p, path_p]))
    except Exception:
        return False


def _build_cmd(img_path: str) -> List[str]:
    # NOTE: inference_streaming.py prompts for image path; we do NOT pass --img_path
    return [
        sys.executable, "-u", "inference_streaming.py",  # -u: unbuffered stdout/stderr
        "--config_path", CONFIG_PATH,
        "--checkpoint_path", CKPT_PATH,
        "--output_folder", OUTPUT_DIR,
        "--seed", SEED,
        "--pretrained_model_path", PRETRAIN_DIR,
    ]


def _send_line(p: subprocess.Popen, text: str):
    if p.stdin:
        try:
            p.stdin.write(text + "
")
            p.stdin.flush()
        except BrokenPipeError:
            _LOGS.append("[server] stdin broken pipe while sending line")
            return
        _LOGS.append(f"[server→child] {text}")
        print("[server→child]", text, flush=True)


def _reader_thread(p: subprocess.Popen):
    global _ready_mouse, _ready_move, _await_image
    for raw in p.stdout:  # type: ignore[arg-type]
        s = raw.rstrip()
        _LOGS.append(s)
        print("[MG2]", s, flush=True)

        low = s.lower()
        if any(tok in low for tok in _IMG_PROMPT_TOKENS):
            with io_lock:
                _await_image = True
                _ready_mouse = False
                _ready_move  = False
            continue

        if "please input the mouse action" in low:
            with io_lock:
                _ready_mouse = True
                _ready_move  = False
            continue

        if ("movement action" in low) or ("press [w, s, a, d, q]" in low):
            with io_lock:
                _ready_move = True
            continue


def _writer_thread(p: subprocess.Popen):
    global _ready_mouse, _ready_move, _await_image
    # Choose how to format the image we send: absolute path, basename, or both
    mode = os.environ.get("MG2_AUTO_IMAGE_MODE", "both").lower()  # abs|name|both
    while True:
        # 1) If the child is asking for an image (or we forced awaiting), send it
        with io_lock:
            awaiting_img = _await_image
            cur_img = _current_image_path
        if awaiting_img:
            try:
                abs_img = str(pathlib.Path(cur_img).resolve())
                base = os.path.basename(abs_img)
                sent_any = False
                if mode in ("abs", "both") and _within(IMAGES_DIR, abs_img) and os.path.exists(abs_img):
                    _send_line(p, abs_img); sent_any = True
                    time.sleep(0.05)
                if mode in ("name", "both"):
                    _send_line(p, base); sent_any = True
                if not sent_any:
                    _LOGS.append(f"[server] Not sending image; path invalid: {abs_img}")
            finally:
                with io_lock:
                    _await_image = False
            time.sleep(0.01)
            continue

        # 2) Otherwise, pull control commands from the queue (mouse, move)
        try:
            item = _in_q.get(timeout=0.1)
        except queue.Empty:
            continue

        if item == "STOP":
            break

        mouse, move = item  # type: ignore[misc]

        # wait for mouse prompt
        while True:
            with io_lock:
                if _ready_mouse:
                    break
            time.sleep(0.01)
        _send_line(p, (mouse or "U"))
        with io_lock:
            _ready_mouse = False
            _ready_move  = True

        # wait for move prompt
        while True:
            with io_lock:
                if _ready_move:
                    break
            time.sleep(0.01)
        _send_line(p, (move or "Q"))
        with io_lock:
            _ready_move = False


reader_t: Optional[threading.Thread] = None
writer_t: Optional[threading.Thread] = None


def _stop_proc():
    global proc, reader_t, writer_t, _ready_mouse, _ready_move, _await_image
    with proc_lock:
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        proc = None
    # reset flags/queue
    with io_lock:
        _ready_mouse = False
        _ready_move  = False
        _await_image = False
    while not _in_q.empty():
        try:
            _in_q.get_nowait()
        except Exception:
            break


def _start_proc(img_path: str):
    global proc, reader_t, writer_t, _current_image_path, _await_image
    if not _within(IMAGES_DIR, img_path):
        raise RuntimeError("Image must be inside IMAGES_DIR")

    cmd = _build_cmd(img_path)
    with proc_lock:
        p = subprocess.Popen(
            cmd, cwd=ROOT_DIR,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=1, text=True,  # line-buffered text I/O
        )
        proc = p
    _current_image_path = img_path

    # spawn fresh I/O threads bound to this proc
    rt = threading.Thread(target=lambda: _reader_thread(p), daemon=True)
    wt = threading.Thread(target=lambda: _writer_thread(p), daemon=True)
    rt.start(); wt.start()
    globals()['reader_t'] = rt
    globals()['writer_t'] = wt

    # proactively mark that we're awaiting image so writer sends immediately
    with io_lock:
        _await_image = True

    # optional extra push: send again shortly after start
    if os.environ.get("MG2_FORCE_IMAGE_ON_BOOT", "1") == "1":
        def _fallback_send():
            time.sleep(0.3)
            with io_lock:
                _await_image = True
        threading.Thread(target=_fallback_send, daemon=True).start()

    # --- fallback: if the child never prints a recognizable image prompt, try sending the
    # selected image once after a short delay (opt-in via env MG2_FORCE_IMAGE_ON_BOOT=1)
    if os.environ.get("MG2_FORCE_IMAGE_ON_BOOT", "0") == "1":
        def _fallback_send():
            time.sleep(2.0)
            with io_lock:
                awaiting = True  # behave as if it asked; safe if it's actually waiting
                img = _current_image_path
            try:
                abs_img = str(pathlib.Path(img).resolve())
                if _within(IMAGES_DIR, abs_img) and os.path.exists(abs_img):
                    _send_line(p, abs_img)
            except Exception as e:
                _LOGS.append(f"[server] fallback send failed: {e}")
        threading.Thread(target=_fallback_send, daemon=True).start()


# start initially
if os.path.exists(_current_image_path):
    _start_proc(_current_image_path)
else:
    _LOGS.append(f"Initial image not found: {_current_image_path}")

# ---------------------------
# Helpers for mp4 serving
# ---------------------------

def _latest_mp4() -> Optional[str]:
    # Prefer *_current.mp4; fallback to any mp4
    mp4s = glob.glob(os.path.join(OUTPUT_DIR, "**", "*_current.mp4"), recursive=True)
    if not mp4s:
        mp4s = glob.glob(os.path.join(OUTPUT_DIR, "**", "*.mp4"), recursive=True)
    if not mp4s:
        return None
    mp4s.sort(key=lambda p: os.path.getmtime(p))
    return mp4s[-1]


def _current_mp4_path() -> Optional[str]:
    return _latest_mp4()


# ---------------------------
# UI (index) with gallery
# ---------------------------
INDEX_HTML = """
<!doctype html>
<title>Matrix-Game 2.0 — Live Control</title>
<style>
  body{font-family:system-ui,sans-serif;margin:1rem}
  .row{display:flex;gap:.5rem;margin:.5rem 0;flex-wrap:wrap;align-items:center}
  button{padding:.5rem .75rem;font-size:1rem}
  video{max-width:100%;border:1px solid #ccc;border-radius:8px}
  img.thumb{width:144px;height:96px;object-fit:cover;border-radius:8px;border:1px solid #ccc;cursor:pointer}
  .muted{color:#666;font-size:.9rem}
  .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(156px,1fr));gap:.75rem}
  .card{display:flex;flex-direction:column;gap:.25rem;align-items:center}
  .sel{border:2px solid #0a84ff !important}
</style>

<h1>Matrix-Game 2.0 — Live Control</h1>

<div class="row muted"><strong>Keys:</strong> Mouse: I K J L U &nbsp;|&nbsp; Move: W A S D Q</div>

<div class="row">
  <label>Mouse:</label>
  <select id="mouse"><option>U</option><option>I</option><option>K</option><option>J</option><option>L</option></select>
  <label>Move:</label>
  <select id="move"><option>Q</option><option>W</option><option>A</option><option>S</option><option>D</option></select>
  <button onclick="sendCmd()">Send</button>
  <button onclick="restart()">Restart</button>
  <span class="muted" id="sel"></span>
</div>

<h3>Images</h3>
<div id="gallery" class="grid"></div>

<h3>Clip</h3>
<video id="vid" controls autoplay playsinline muted></video>
<div class="row muted" id="status"></div>

<script>
const vid = document.getElementById('vid');
const statusEl = document.getElementById('status');
const gallery = document.getElementById('gallery');
const sel = document.getElementById('sel');
let lastTag = null;
let currentImg = null;

async function sendCmd(){
  const mouse=document.getElementById('mouse').value;
  const move=document.getElementById('move').value;
  await fetch('/cmd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mouse,move})});
}

async function restart(imgPath){
  // If imgPath omitted, reuse current
  const body = imgPath ? {img: imgPath, purge:true} : {purge:false};
  const r = await fetch('/restart',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const j = await r.json();
  if(j.ok){
    currentImg = j.image;
    updateSel();
    // Show the selected image as a poster until the first clip appears
    if (currentImg) {
      vid.removeAttribute('src');
      vid.setAttribute('poster', `/image?name=${currentImg}`);
      vid.load();
    }
    lastTag = null; // force video reload on next /meta tick
  }
}

function updateSel(){
  sel.textContent = currentImg ? ('Selected: ' + currentImg) : '';
  document.querySelectorAll('.thumb').forEach(el=>{
    if(currentImg && el.getAttribute('data-name')===currentImg) el.classList.add('sel');
    else el.classList.remove('sel');
  });
}

async function loadImages(){
  const r = await fetch('/images');
  const j = await r.json();
  gallery.innerHTML = '';
  j.items.forEach(it => {
    const card = document.createElement('div');
    card.className = 'card';
    const img = document.createElement('img');
    img.className = 'thumb';
    img.src = it.thumb;
    img.title = it.name;
    img.setAttribute('data-name', it.name);
    img.onclick = () => restart(it.path);
    const cap = document.createElement('div');
    cap.className = 'muted';
    cap.textContent = it.name;
    card.appendChild(img); card.appendChild(cap);
    gallery.appendChild(card);
  });
  currentImg = j.selected || null;
  updateSel();
  // If there is no video yet, show the selected image as poster on initial load
  if (currentImg) {
    vid.removeAttribute('src');
    vid.setAttribute('poster', `/image?name=${currentImg}`);
    vid.load();
  }
}

// Poll for new 1s clip; reload <video> when mtime/size changes
async function refreshIfChanged(){
  try{
    const r = await fetch('/meta');
    const j = await r.json();
    if(!j.exists){
      statusEl.textContent='Waiting for first clip…';
      // Ensure poster shows the current selection while we wait
      if (j.selected_url) {
        vid.removeAttribute('src');
        vid.setAttribute('poster', `${j.selected_url}?t=${encodeURIComponent(j.selected||'')}`);
        vid.load();
      }
      return;
    }
    const tag = `${j.mtime}-${j.size}`;
    statusEl.textContent = `Current: ${j.path}  |  size: ${j.size}  |  mtime: ${new Date(j.mtime*1000).toLocaleTimeString()}`;
    if(tag !== lastTag){
      lastTag = tag;
      const url = `/current.mp4?t=${encodeURIComponent(tag)}`;
      const wasPaused = vid.paused;
      vid.removeAttribute('poster');
      vid.src = url;
      await vid.play().catch(()=>{});
      if (wasPaused) vid.pause();
    }
  }catch(e){}
}

setInterval(refreshIfChanged, 250);
refreshIfChanged();
loadImages();
</script>
"""

# ---------------------------
# Routes
# ---------------------------
@APP.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@APP.get("/favicon.ico")
def favicon():
    return PlainTextResponse("", status_code=204)


@APP.get("/healthz")
def health():
    with proc_lock:
        alive = (proc is not None and proc.poll() is None)
    cur = _current_mp4_path()
    return {"proc_alive": alive, "latest_mp4": cur, "output_dir": OUTPUT_DIR, "image": _current_image_path}


@APP.get("/logs")
def logs():
    return JSONResponse({"lines": list(_LOGS)})


# --- Images listing and thumbnails ---

def _list_images() -> List[str]:
    pats = ["*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"]
    files: List[str] = []
    for pat in pats:
        files += glob.glob(os.path.join(IMAGES_DIR, pat))
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files


@APP.get("/images")
def images():
    items = []
    for p in _list_images():
        name = os.path.basename(p)
        items.append({
            "name": name,
            "path": str(pathlib.Path(p).resolve()),
            "thumb": f"/thumb?name={name}",
            "url": f"/image?name={name}"
        })
    sel = os.path.basename(_current_image_path) if _within(IMAGES_DIR, _current_image_path) else None
    return {"items": items, "selected": sel, "dir": IMAGES_DIR}


@APP.get("/image")
def image(name: str = Query(...)):
    path = str(pathlib.Path(IMAGES_DIR, name).resolve())
    if not _within(IMAGES_DIR, path) or not os.path.exists(path):
        raise HTTPException(404)
    with open(path, "rb") as f:
        data = f.read()
    # naive content-type; browser only needs it for viewing
    return Response(content=data, media_type="image/*", headers={"Cache-Control": "no-store"})


@APP.get("/thumb")
def thumb(name: str = Query(...), w: int = 256, h: int = 160):
    path = str(pathlib.Path(IMAGES_DIR, name).resolve())
    if not _within(IMAGES_DIR, path) or not os.path.exists(path):
        raise HTTPException(404)
    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((w, h))
        bio = io.BytesIO()
        img.save(bio, format="JPEG", quality=85, optimize=True)
        return Response(content=bio.getvalue(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})
    except Exception:
        raise HTTPException(500)


# --- Restart with (optional) new image ---
@APP.post("/restart")
async def restart(req: Request):
    body = await req.json()
    img = body.get("img")  # absolute path (from /images), or None to reuse current
    purge = bool(body.get("purge", False))

    global _current_image_path
    if img:
        # validate
        if not _within(IMAGES_DIR, img) or not os.path.exists(img):
            raise HTTPException(400, "Image must be inside IMAGES_DIR")
        _current_image_path = img

    # purge outputs if requested
    if purge and os.path.exists(OUTPUT_DIR):
        for p in glob.glob(os.path.join(OUTPUT_DIR, "*")):
            try:
                if os.path.isdir(p): shutil.rmtree(p)
                else: os.remove(p)
            except Exception:
                pass

    # stop & start
    _stop_proc()
    _start_proc(_current_image_path)
    return JSONResponse({"ok": True, "image": os.path.basename(_current_image_path)})


# --- Control commands ---
@APP.post("/cmd")
async def cmd_endpoint(req: Request):
    data = await req.json()
    mouse = (data.get("mouse") or "U").upper()[0]
    move  = (data.get("move")  or "Q").upper()[0]
    _in_q.put((mouse, move))
    return JSONResponse({"ok": True, "mouse": mouse, "move": move})


# --- MP4 metadata & streaming with HTTP Range ---

def _meta_payload():
    p = _current_mp4_path()
    # always include the currently selected image so the client can show a poster
    sel_name = os.path.basename(_current_image_path) if _within(IMAGES_DIR, _current_image_path) else None
    sel_url = f"/image?name={sel_name}" if sel_name else None
    if not p or not os.path.exists(p):
        return {"exists": False, "selected": sel_name, "selected_url": sel_url}
    return {
        "exists": True,
        "path": os.path.basename(p),
        "mtime": os.path.getmtime(p),
        "size": os.path.getsize(p),
        "selected": sel_name,
        "selected_url": sel_url,
    }


@APP.get("/meta")
def meta():
    return _meta_payload()


@APP.get("/current.mp4")
def current_mp4(range: str | None = Header(default=None)):
    path = _current_mp4_path()
    if not path or not os.path.exists(path):
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
        start = max(0, start); end = min(end, file_size - 1)

    def _read():
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            chunk = 1024 * 1024
            while remaining > 0:
                data = f.read(min(chunk, remaining))
                if not data: break
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