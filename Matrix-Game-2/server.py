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
    "input the image path", "image path:", "please input the image",
    "please input image", "enter image path", "path of the image",
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
        # Ensure a newline so the child process receives a full line and proceeds.
        p.stdin.write(text + "\n")
        p.stdin.flush()
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
    while True:
        # 1) If the child is asking for an image path, immediately send our current one
        with io_lock:
            awaiting_img = _await_image
            cur_img = _current_image_path
        if awaiting_img:
            # validate the path and send the absolute path
            try:
                abs_img = str(pathlib.Path(cur_img).resolve())
                if not _within(IMAGES_DIR, abs_img) or not os.path.exists(abs_img):
                    _LOGS.append(f"[server] Image not found or outside IMAGES_DIR: {abs_img}")
                else:
                    _send_line(p, abs_img)
            finally:
                with io_lock:
                    _await_image = False
            # loop to catch subsequent prompts quickly
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
    global proc, reader_t, writer_t, _current_image_path
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
    :root{
        --bg:#0b0d10; --panel:#11151a; --panel2:#151a21; --text:#e6edf3; --muted:#9aa7b2;
        --accent:#3ea6ff; --accent2:#00d1b2; --border:#222a33; --key:#1c222b; --keyText:#e6edf3;
    }
    *{box-sizing:border-box}
    html,body{height:100%}
    body{margin:0;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Helvetica Neue",Arial;background:radial-gradient(1200px 800px at 50% -20%, #1a2330 0%, #0b0d10 60%);color:var(--text)}
    a{color:var(--accent)}

    .wrap{min-height:100%;display:flex;flex-direction:column}
    header{padding:16px 24px;border-bottom:1px solid var(--border);background:linear-gradient(180deg, rgba(255,255,255,0.02), transparent)}
    header .title{display:flex;gap:12px;align-items:center}
    .badge{font-size:12px;padding:3px 8px;border:1px solid var(--border);border-radius:999px;color:var(--muted);background:rgba(0,0,0,.25)}

    main{padding:20px}
    .layout{display:grid;grid-template-columns:1fr;gap:18px;max-width:1400px;margin:0 auto}
    @media (min-width: 1200px){ .layout{grid-template-columns:2fr 1fr} }

    /* Player */
    .player-card{background:var(--panel);border:1px solid var(--border);border-radius:16px;padding:16px;box-shadow:0 10px 30px rgba(0,0,0,.35)}
    .player{position:relative;aspect-ratio:16/9;border-radius:12px;overflow:hidden;background:linear-gradient(180deg, #0d1117, #0b0d10)}
    .player video{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;background:#000}
    .player img.preview{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000;display:none}
    .player .overlay{position:absolute;inset:0;display:flex;align-items:end;justify-content:space-between;padding:12px;background:linear-gradient( to top, rgba(0,0,0,.45), rgba(0,0,0,0) 40%);pointer-events:none}
    .status{font-size:12px;color:var(--muted)}

    /* Controls */
    .controls-card{background:var(--panel2);border:1px solid var(--border);border-radius:16px;padding:16px;display:flex;flex-direction:column;gap:16px}
    .row{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
    .spacer{flex:1}
    button.primary{background:var(--accent);color:#071018;border:none;border-radius:10px;padding:10px 14px;font-weight:600;cursor:pointer}
    button.ghost{background:transparent;color:var(--text);border:1px solid var(--border);border-radius:10px;padding:10px 14px;cursor:pointer}
    button.primary:hover{filter:brightness(1.05)}
    button.ghost:hover{border-color:#334153}

    .keypad{display:flex;gap:22px;flex-wrap:wrap}
    .pad{display:grid;grid-template-columns:repeat(3,56px);grid-auto-rows:56px;gap:8px;align-items:center;justify-content:center;background:linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.01));border:1px solid var(--border);border-radius:14px;padding:12px}
    .pad-title{grid-column:1/-1;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;justify-self:start}
    .key{display:flex;align-items:center;justify-content:center;border-radius:10px;background:var(--key);color:var(--keyText);box-shadow:inset 0 1px 0 rgba(255,255,255,.04), 0 4px 14px rgba(0,0,0,.35);border:1px solid #28303a;cursor:pointer;user-select:none;font-weight:700}
    .key.small{font-size:12px}
    .key:active{transform:translateY(1px)}
    .hint{font-size:12px;color:var(--muted)}

    /* Gallery */
    .gallery-card{background:var(--panel2);border:1px solid var(--border);border-radius:16px;padding:16px}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px}
    .thumb-wrap{position:relative;border-radius:12px;overflow:hidden;border:1px solid var(--border);background:#0b0f14}
    img.thumb{width:100%;height:110px;object-fit:cover;display:block;opacity:.95;transition:opacity .15s ease, transform .15s ease}
    .thumb-wrap:hover img.thumb{opacity:1;transform:scale(1.01)}
    .thumb-name{font-size:12px;color:var(--muted);margin-top:6px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .sel{outline:2px solid var(--accent)}

    /* Hidden native selectors, kept for fallback */
    .native-selects{display:none}
    @media (prefers-reduced-motion:no-preference){ .pulse{animation:pulse 1.2s ease-in-out infinite} }
    @keyframes pulse{0%{opacity:.6}50%{opacity:1}100%{opacity:.6}}
</style>

<div class="wrap">
    <header>
        <div class="title">
            <h2 style="margin:0">Matrix-Game 2.0 — Live Control</h2>
            <span class="badge">Streaming</span>
        </div>
    </header>

    <main>
        <div class="layout">
            <section class="player-card">
                <div class="player" id="player">
                    <video id="vid" controls autoplay playsinline muted></video>
                    <img id="preview" class="preview" alt="Selected image preview" />
                    <div class="overlay">
                        <div class="status" id="status">Initializing…</div>
                        <div class="status">Tip: Use the on-screen pads or your keyboard</div>
                    </div>
                </div>
            </section>

            <aside class="controls-card">
                <div class="row">
                    <div class="hint">Select an image below, then use the pads to look and move.</div>
                    <div class="spacer"></div>
                    <button class="ghost" onclick="restart()">Restart</button>
                </div>

                <div class="keypad">
                    <div class="pad" id="lookPad">
                        <div class="pad-title">Look</div>
                        <div></div>
                        <div class="key" data-m="I" data-v="Q">I</div>
                        <div></div>
                        <div class="key" data-m="J" data-v="Q">J</div>
                        <div class="key small" data-m="U" data-v="Q" title="No look">•</div>
                        <div class="key" data-m="L" data-v="Q">L</div>
                        <div></div>
                        <div class="key" data-m="K" data-v="Q">K</div>
                        <div></div>
                    </div>

                    <div class="pad" id="movePad">
                        <div class="pad-title">Move</div>
                        <div></div>
                        <div class="key" data-m="U" data-v="W">W</div>
                        <div></div>
                        <div class="key" data-m="U" data-v="A">A</div>
                        <div class="key" data-m="U" data-v="Q" title="Stop">■</div>
                        <div class="key" data-m="U" data-v="D">D</div>
                        <div></div>
                        <div class="key" data-m="U" data-v="S">S</div>
                        <div></div>
                    </div>
                </div>

                <div class="row native-selects">
                    <label>Mouse:</label>
                    <select id="mouse"><option>U</option><option>I</option><option>K</option><option>J</option><option>L</option></select>
                    <label>Move:</label>
                    <select id="move"><option>Q</option><option>W</option><option>A</option><option>S</option><option>D</option></select>
                    <button class="primary" onclick="sendCmd()">Send</button>
                </div>

                <div class="row"><span id="sel" class="hint"></span></div>
            </aside>
        </div>

        <section class="gallery-card">
            <h3 style="margin:0 0 10px 0">Images</h3>
            <div id="gallery" class="grid"></div>
        </section>
    </main>
</div>

<script>
const vid = document.getElementById('vid');
const previewImg = document.getElementById('preview');
const statusEl = document.getElementById('status');
const gallery = document.getElementById('gallery');
const sel = document.getElementById('sel');
let lastTag = null;
let currentImg = null;
let activityStarted = false; // true once user looks or moves

// Lightweight helpers
function showPreview(url){
    previewImg.src = url;
    previewImg.style.display = 'block';
    vid.poster = url;
}
function hidePreview(){
    previewImg.style.display = 'none';
}

async function send(mouse, move){
    await fetch('/cmd',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mouse,move})});
}

async function sendCmd(){
    const mouse=document.getElementById('mouse').value;
    const move=document.getElementById('move').value;
    if(mouse!=='U' || move!=='Q') activityStarted = true;
    await send(mouse, move);
}

async function restart(imgPath){
    // If imgPath omitted, reuse current
    const body = imgPath ? {img: imgPath, purge:true} : {purge:false};
    const r = await fetch('/restart',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j = await r.json();
    if(j.ok){
        currentImg = j.image;
        updateSel();
        // show the selected image as preview until activity starts and video is ready
        const url = `/image?name=${encodeURIComponent(currentImg)}`;
        showPreview(url);
        lastTag = null; // force video reload on next /meta tick
        activityStarted = false;
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
        const wrap = document.createElement('div');
        wrap.className = 'thumb-wrap';
        const img = document.createElement('img');
        img.className = 'thumb';
        img.src = it.thumb;
        img.title = it.name;
        img.setAttribute('data-name', it.name);
        img.onclick = () => restart(it.path);
        wrap.appendChild(img);
        const cap = document.createElement('div');
        cap.className = 'thumb-name';
        cap.textContent = it.name;
        const cont = document.createElement('div');
        cont.appendChild(wrap); cont.appendChild(cap);
        gallery.appendChild(cont);
    });
    currentImg = j.selected || null;
    updateSel();
    if(currentImg){
        const url = `/image?name=${encodeURIComponent(currentImg)}`;
        showPreview(url);
    }
}

// Wire gamepad clicks
function setupPads(){
    document.querySelectorAll('.pad .key').forEach(k=>{
        k.addEventListener('click', async ()=>{
            const m = k.getAttribute('data-m') || 'U';
            const v = k.getAttribute('data-v') || 'Q';
            if(m!=='U' || v!=='Q') activityStarted = true;
            await send(m, v);
        });
    });
}

// Optional keyboard support (IJKLU, WASDQ)
const keyMap = {
    'i': ['I','Q'], 'k': ['K','Q'], 'j': ['J','Q'], 'l': ['L','Q'], 'u': ['U','Q'],
    'w': ['U','W'], 'a': ['U','A'], 's': ['U','S'], 'd': ['U','D'], 'q': ['U','Q']
};
window.addEventListener('keydown', async (ev)=>{
    const k = ev.key.toLowerCase();
    if(keyMap[k]){
        ev.preventDefault();
        const [m,v] = keyMap[k];
        if(m!=='U' || v!=='Q') activityStarted = true;
        await send(m,v);
    }
});

// Handle video state and auto-switch from preview when stream updates
vid.addEventListener('playing', ()=>{
    if(activityStarted) hidePreview();
});
vid.addEventListener('loadeddata', ()=>{
    // In case autoplay is blocked, still hide preview once loaded and activity started
    if(activityStarted && !vid.paused) hidePreview();
});

// Poll for new 1s clip; reload <video> when mtime/size changes
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
            vid.src = url;
            await vid.play().catch(()=>{});
            if (wasPaused) vid.pause();
            // if activity already started, attempt to hide preview soon
            if(activityStarted){
                setTimeout(()=>hidePreview(), 200);
            }
        }
    }catch(e){}
}

setInterval(refreshIfChanged, 350);
refreshIfChanged();
loadImages();
setupPads();
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
    if not p or not os.path.exists(p):
        return {"exists": False}
    return {
        "exists": True,
        "path": os.path.basename(p),
        "mtime": os.path.getmtime(p),
        "size": os.path.getsize(p),
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