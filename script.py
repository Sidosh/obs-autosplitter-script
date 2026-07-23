"""OBS autosplitter script.

On load, starts the OBS Virtual Camera and determines which cv2 (OpenCV)
device index it is reachable under, so frames can later be captured with
cv2.VideoCapture(camera_index, cv2.CAP_DSHOW).

Also keeps a WebSocket connection to the LiveSplit server open
(ws://<host>:<port>/livesplit, default port 16834) and reconnects
automatically, used for the "Test LiveSplit connection" button
(livesplit_send()). The WS server must be started inside LiveSplit first:
right click LiveSplit -> Control -> Start WS Server. Actual splits are
sent over their own connection by the --autosplit subprocess (see below).

Dependencies (opencv-python, pygrabber, websocket-client) are installed
automatically into the user site-packages of OBS's configured Python
when missing. Manual install, if ever needed:
    C:\Python311\python.exe -m pip install --user opencv-python pygrabber websocket-client
pygrabber finds the camera by its DirectShow device name; without it the
script falls back to probing every index.

The actual autosplitting (camera reads + template matching) runs in a
separate `python.exe --autosplit ...` subprocess of this same file rather
than on a thread inside OBS: cv2.VideoCapture(CAP_DSHOW) only receives
real frames from the OBS Virtual Camera when opened from outside OBS's
own process, so a reader living inside OBS's process gets nothing but
blank frames. See standalone_autosplit_main() and restart_autosplit_process().
"""

import ctypes
import importlib
import importlib.util
import json
import os
import site
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import quote

try:
    import obspython as obs
    OBS_LOG_LEVELS = {"INFO": obs.LOG_INFO, "WARNING": obs.LOG_WARNING, "ERROR": obs.LOG_ERROR}
except ImportError:
    obs = None  # running standalone as the --autosplit subprocess
    OBS_LOG_LEVELS = {}

RETRY_INTERVAL_MS = 1000
MAX_START_ATTEMPTS = 15
PROBE_MAX_INDEX = 10
PIP_TIMEOUT_S = 600
LIVESPLIT_RETRY_S = 5
LIVESPLIT_CONNECT_TIMEOUT_S = 5
# recv() timeout; also how quickly the worker notices a stop request.
LIVESPLIT_RECV_TIMEOUT_S = 1
# How often the --autosplit subprocess polls LiveSplit's timer phase to
# notice a reset it did not itself trigger (see livesplit_reset_watcher).
RESET_POLL_INTERVAL_S = 1

SCRIPT_PATH = os.path.abspath(__file__)
SCRIPT_DIR = os.path.dirname(SCRIPT_PATH)
GAMES_DIR = os.path.join(SCRIPT_DIR, "games")
SPLITS_FILENAME = "splits.json"

# cv2.matchTemplate (TM_CCOEFF_NORMED, or its masked equivalent for
# templates with real PNG transparency - see prepare_masked_template)
# score above which a template counts as detected in the current frame;
# tune per game if splits misfire.
MATCH_THRESHOLD = 0.85
# A template's own pixel std (0-255 scale) below this counts as "flat"
# (e.g. a plain solid-color reference image, like a white flash screen).
# TM_CCOEFF_NORMED normalizes by the template's own variance, so a flat
# template makes cv2 short-circuit to a score of 1.0 for every position
# regardless of what's actually in the frame - flat templates are matched
# by direct pixel-color comparison instead (see flat_match_score).
FLAT_TEMPLATE_STD = 2.0
# An "images" list entry in splits.json may be either a filename (matched
# against the full frame) or {filename: [x, y, w, h]} (pixels, in the
# camera frame's own coordinate space) to restrict matchTemplate to that
# rect for just that image. Narrows the search space (faster) and ignores
# unrelated on-screen changes (fewer false positives).
# How often the autosplitter re-reads the camera while a split's images
# haven't matched yet.
FRAME_POLL_INTERVAL_S = 0.01
# How often a non-matching score is logged while waiting on a split image,
# so misfires can be diagnosed from the script log without needing to
# reproduce them under helper/preview_matches.py.
SCORE_LOG_INTERVAL_S = 2
# How often the autosplitter checks whether the camera index / the
# selected category's local files have become available.
AUTOSPLIT_WAIT_S = 1

# Games bundled in the script's own repo are also offered even when not
# downloaded yet; picking one fetches it into GAMES_DIR for local/offline use.
GITHUB_OWNER = "Sidosh"
GITHUB_REPO = "obs-autosplitter-script"
GITHUB_BRANCH = "main"
GITHUB_API_BASE = f"https://api.github.com/repos/{GITHUB_OWNER}/{GITHUB_REPO}/contents"
GITHUB_RAW_BASE = f"https://raw.githubusercontent.com/{GITHUB_OWNER}/{GITHUB_REPO}/{GITHUB_BRANCH}"
GITHUB_TIMEOUT_S = 5
# Avoids hitting GitHub's unauthenticated rate limit when the properties
# dialog is reopened or the game/category dropdowns are toggled repeatedly.
GITHUB_CACHE_TTL_S = 60

# import name -> pip package name
REQUIRED_PACKAGES = {
    "cv2": "opencv-python",
    "pygrabber": "pygrabber",
    "websocket": "websocket-client",
}

camera_index = -1
start_attempts = 0
started_by_script = False
unloading = False
detect_thread = None
deps_lock = threading.Lock()

livesplit_url = ""
livesplit_thread = None
livesplit_stop = None
livesplit_lock = threading.Lock()
livesplit_conn = None  # connected WebSocket, guarded by livesplit_lock

# path (relative to the repo root) -> (fetched_at, GitHub API 'contents' entries)
_github_contents_cache = {}

# game/category currently armed in the autosplit subprocess (script_update
# restarts it when these, or the LiveSplit URL, no longer match the settings).
autosplit_game = ""
autosplit_category = ""
autosplit_process = None  # subprocess.Popen of `python.exe script.py --autosplit ...`
autosplit_reader_thread = None

# Settings from the last script_load()/script_update(), kept so
# script_properties() can populate the category list for the game that is
# already selected when the properties dialog is (re)opened.
current_settings = None


def get_camera_index():
    """cv2 device index of the OBS Virtual Camera, or -1 if not found."""
    return camera_index


def log(level, msg):
    """Logs `msg` at `level` ("INFO"/"WARNING"/"ERROR").

    Inside OBS, goes to obs.script_log(). Standalone (the --autosplit
    subprocess, where `obs` is None), goes to stdout instead, prefixed with
    the level, so read_autosplit_output() in the OBS-side process can
    relay it back into the script log at the right level.
    """
    if obs is None:
        print(f"{level}: {msg}", flush=True)
        return
    if not unloading:
        obs.script_log(OBS_LOG_LEVELS[level], msg)


def ensure_dependencies():
    """Install missing packages via pip into the user site-packages.

    Runs on a worker thread, so OBS stays responsive during the
    download/install. deps_lock keeps concurrent workers (camera
    detection, LiveSplit connection) from racing pip.
    """
    with deps_lock:
        missing = [pkg for mod, pkg in REQUIRED_PACKAGES.items()
                   if importlib.util.find_spec(mod) is None]
        if not missing:
            return True

        # In OBS's embedded interpreter sys.executable is obs64.exe, so pip
        # must be run through the configured Python install (sys.base_prefix).
        python_exe = os.path.join(sys.base_prefix, "python.exe")
        if not os.path.isfile(python_exe):
            log("ERROR",
                f"Cannot auto-install {missing}: python.exe not found in "
                f"{sys.base_prefix}. Install manually with: "
                f"pip install --user {' '.join(missing)}")
            return False

        log("INFO",
            f"Installing missing packages: {', '.join(missing)} "
            "(this can take a minute)...")
        try:
            # --user: the Python install dir is usually not writable without
            # admin rights.
            result = subprocess.run(
                [python_exe, "-m", "pip", "install", "--user", *missing],
                capture_output=True, text=True, timeout=PIP_TIMEOUT_S,
                creationflags=subprocess.CREATE_NO_WINDOW)
        except (OSError, subprocess.TimeoutExpired) as e:
            log("ERROR", f"Could not run pip install: {e}")
            return False
        if result.returncode != 0:
            tail = (result.stderr or result.stdout
                    or "").strip().splitlines()[-5:]
            log("ERROR", "pip install failed:\n" + "\n".join(tail))
            return False

        # A fresh --user install may land in a directory that was not on
        # sys.path when the interpreter started.
        user_site = site.getusersitepackages()
        if user_site not in sys.path:
            sys.path.insert(0, user_site)
        importlib.invalidate_caches()
        log("INFO", f"Installed {', '.join(missing)}")
        return True


def find_index_by_name():
    # pygrabber enumerates DirectShow devices in the same order cv2 uses
    # with the CAP_DSHOW backend, so the list position is the cv2 index.
    try:
        from pygrabber.dshow_graph import FilterGraph

        devices = FilterGraph().get_input_devices()
    except ImportError:
        log("WARNING",
            "pygrabber not installed, falling back to probing indices "
            "by resolution (pip install pygrabber)")
        return -1
    except OSError as e:
        log("WARNING",
            f"pygrabber device enumeration failed ({e}), falling back "
            "to probing indices by resolution")
        return -1

    for i, name in enumerate(devices):
        if "OBS Virtual Camera" in name:
            log("INFO",
                f"Found '{name}' at cv2 index {i} (by device name)")
            return i
    log("WARNING", f"OBS Virtual Camera not in device list: {devices}")
    return -1


def find_index_by_resolution():
    # Fallback: open each index and compare its resolution to the OBS
    # output resolution, which is what the virtual camera outputs.
    try:
        import cv2
    except ImportError:
        log("ERROR",
            "opencv-python is not installed in OBS's Python "
            "(pip install opencv-python)")
        return -1

    ovi = obs.obs_video_info()
    obs.obs_get_video_info(ovi)
    target = (ovi.output_width, ovi.output_height)

    for i in range(PROBE_MAX_INDEX):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
        if size == target:
            log("INFO",
                f"Found virtual camera at cv2 index {i} "
                f"(matched output resolution {target[0]}x{target[1]})")
            return i
    return -1


def detection_worker():
    """Runs on its own thread so COM apartment state is fully ours.

    OBS's callback threads may already have COM initialized in either STA
    or MTA mode (it varies between launches), and a thread's mode cannot
    be changed once set (RPC_E_CHANGED_MODE). A fresh Python thread has no
    mode yet, so initializing STA here always succeeds and pygrabber/cv2
    get a predictable COM environment.
    """
    global camera_index

    ensure_dependencies()

    sys.coinit_flags = 2  # COINIT_APARTMENTTHREADED; comtypes reads this
    hr = ctypes.windll.ole32.CoInitializeEx(None, 2)
    try:
        index = find_index_by_name()
        if index < 0:
            index = find_index_by_resolution()
    finally:
        if hr >= 0:
            ctypes.windll.ole32.CoUninitialize()

    camera_index = index
    if index < 0:
        log("ERROR", "Could not find the OBS Virtual Camera index")


def start_detection():
    global detect_thread
    if detect_thread is not None and detect_thread.is_alive():
        return
    detect_thread = threading.Thread(target=detection_worker, daemon=True)
    detect_thread.start()


def startup_tick():
    """Runs every RETRY_INTERVAL_MS until the virtual camera is up."""
    global start_attempts, started_by_script

    if not obs.obs_frontend_virtualcam_active():
        start_attempts += 1
        if start_attempts > MAX_START_ATTEMPTS:
            obs.timer_remove(startup_tick)
            log("ERROR",
                "Giving up: virtual camera did not start after "
                f"{MAX_START_ATTEMPTS} attempts")
            return
        obs.obs_frontend_start_virtualcam()
        started_by_script = True
        return  # verify it is active on the next tick

    obs.timer_remove(startup_tick)
    log("INFO", "Virtual camera is active")
    start_detection()


def redetect_clicked(props, prop):
    start_detection()
    return False


# --- LiveSplit WebSocket connection ------------------------------------------

def livesplit_send(command):
    """Send a timer command (e.g. 'startorsplit', 'split') to LiveSplit.

    Returns True if the command was written to the socket. Responses to
    query commands (e.g. 'getcurrenttimerphase') arrive asynchronously
    and are written to the script log by the receive loop.
    """
    with livesplit_lock:
        conn = livesplit_conn
    if conn is None:
        log("WARNING",
            f"Not connected to LiveSplit, dropped command '{command}'")
        return False
    try:
        conn.send(command)
        return True
    except Exception as e:
        log("WARNING", f"Sending '{command}' to LiveSplit failed: {e}")
        return False


def livesplit_worker(url, stop):
    """Keeps a WebSocket connection to the LiveSplit server alive.

    Connects and reconnects until `stop` is set (script unload or a
    settings change, which starts a fresh worker for the new URL).
    """
    global livesplit_conn

    if not ensure_dependencies():
        return
    try:
        import websocket
    except ImportError:
        log("ERROR",
            "websocket-client is not installed in OBS's Python "
            "(pip install --user websocket-client)")
        return

    failure_logged = False
    while not stop.is_set():
        try:
            conn = websocket.create_connection(
                url, timeout=LIVESPLIT_CONNECT_TIMEOUT_S)
        except Exception as e:
            if not failure_logged:
                log("WARNING",
                    f"Cannot reach LiveSplit at {url} ({e}). Make sure the "
                    "WS server is running: right click LiveSplit -> Control "
                    "-> Start WS Server. Retrying every "
                    f"{LIVESPLIT_RETRY_S}s...")
                failure_logged = True
            stop.wait(LIVESPLIT_RETRY_S)
            continue

        failure_logged = False
        conn.settimeout(LIVESPLIT_RECV_TIMEOUT_S)
        with livesplit_lock:
            livesplit_conn = conn
        log("INFO", f"Connected to LiveSplit at {url}")

        try:
            while not stop.is_set():
                try:
                    msg = conn.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if msg:
                    log("INFO", f"LiveSplit: {msg}")
                elif not conn.connected:  # server sent a close frame
                    break
        except Exception:
            pass  # connection dropped; outer loop reconnects
        finally:
            with livesplit_lock:
                if livesplit_conn is conn:
                    livesplit_conn = None
            try:
                conn.close()
            except Exception:
                pass
        if not stop.is_set():
            log("WARNING", "LiveSplit connection lost, reconnecting...")


def close_livesplit_conn():
    with livesplit_lock:
        conn = livesplit_conn
    if conn is not None:
        try:
            conn.close()
        except Exception:
            pass


def restart_livesplit_worker():
    """Start the connection worker, stopping the previous one first."""
    global livesplit_thread, livesplit_stop

    if livesplit_stop is not None:
        livesplit_stop.set()
    close_livesplit_conn()  # unblock the old worker's recv() immediately

    livesplit_stop = threading.Event()
    livesplit_thread = threading.Thread(
        target=livesplit_worker, args=(livesplit_url, livesplit_stop),
        daemon=True)
    livesplit_thread.start()


def test_livesplit_clicked(props, prop):
    if livesplit_send("getcurrenttimerphase"):
        log("INFO",
            "Sent 'getcurrenttimerphase'; the response appears in this log")
    return False


# --- Game / category selection ------------------------------------------------

def list_subdirs(path):
    """Names of the immediate subdirectories of `path`, sorted; [] if missing."""
    try:
        entries = os.listdir(path)
    except OSError:
        return []
    return sorted((name for name in entries if os.path.isdir(os.path.join(path, name))),
                  key=str.casefold)


def github_contents(path):
    """GitHub API 'contents' listing for `path` in the repo, cached briefly.

    Returns [] on any failure (offline, rate-limited, path doesn't exist)
    so the game/category dropdowns still work with local-only folders.
    """
    now = time.monotonic()
    cached = _github_contents_cache.get(path)
    if cached is not None and now - cached[0] < GITHUB_CACHE_TTL_S:
        return cached[1]

    url = f"{GITHUB_API_BASE}/{quote(path)}?ref={GITHUB_BRANCH}"
    try:
        with urllib.request.urlopen(url, timeout=GITHUB_TIMEOUT_S) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, OSError, ValueError) as e:
        log("WARNING", f"Could not list GitHub path '{path}' ({e})")
        data = []
    entries = data if isinstance(data, list) else []
    _github_contents_cache[path] = (now, entries)
    return entries


def github_list_dirnames(path):
    return [entry["name"] for entry in github_contents(path)
            if entry.get("type") == "dir"]


def github_list_filenames(path):
    return [entry["name"] for entry in github_contents(path)
            if entry.get("type") == "file"]


def list_games():
    names = set(list_subdirs(GAMES_DIR)) | set(github_list_dirnames("games"))
    return sorted(names, key=str.casefold)


def list_categories(game):
    if not game:
        return []
    local = list_subdirs(os.path.join(GAMES_DIR, game))
    remote = github_list_dirnames(f"games/{game}")
    return sorted(set(local) | set(remote), key=str.casefold)


def fill_game_list(prop):
    obs.obs_property_list_clear(prop)
    for name in list_games():
        obs.obs_property_list_add_string(prop, name, name)


def fill_category_list(prop, game):
    obs.obs_property_list_clear(prop)
    for name in list_categories(game):
        obs.obs_property_list_add_string(prop, name, name)


def category_exists_locally(game, category):
    return os.path.isdir(os.path.join(GAMES_DIR, game, category))


def github_download_file(remote_path, local_path):
    url = f"{GITHUB_RAW_BASE}/{quote(remote_path)}"
    try:
        with urllib.request.urlopen(url, timeout=GITHUB_TIMEOUT_S) as resp:
            data = resp.read()
    except (urllib.error.URLError, OSError) as e:
        log("WARNING", f"Could not download '{remote_path}' from GitHub ({e})")
        return False
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        with open(local_path, "wb") as f:
            f.write(data)
    except OSError as e:
        log("WARNING", f"Could not save '{local_path}' ({e})")
        return False
    return True


def download_category_worker(game, category):
    """Fetches one category's splits.json + images from GitHub into GAMES_DIR.

    Runs on its own thread so the properties dialog doesn't block on
    network I/O; afterwards the category is a normal local one.
    """
    remote_base = f"games/{game}/{category}"
    local_base = os.path.join(GAMES_DIR, game, category)

    ok = github_download_file(f"{remote_base}/{SPLITS_FILENAME}",
                              os.path.join(local_base, SPLITS_FILENAME))
    for name in github_list_filenames(f"{remote_base}/images"):
        ok = github_download_file(f"{remote_base}/images/{name}",
                                  os.path.join(local_base, "images", name)) and ok

    if ok:
        log("INFO", f"Downloaded '{game}/{category}' from GitHub")
    else:
        log("WARNING",
            f"'{game}/{category}' download from GitHub finished with errors")


def download_category_if_missing(game, category):
    if not game or not category or category_exists_locally(game, category):
        return
    log("INFO", f"'{game}/{category}' not found locally, downloading from GitHub...")
    threading.Thread(target=download_category_worker, args=(game, category),
                     daemon=True).start()


def category_changed(props, prop, settings):
    """Fetches the selected category from GitHub if not already local."""
    game = obs.obs_data_get_string(settings, "game")
    category = obs.obs_data_get_string(settings, "category")
    download_category_if_missing(game, category)
    return False


def game_changed(props, prop, settings):
    """Refreshes the category list to match the newly selected game."""
    game = obs.obs_data_get_string(settings, "game")
    fill_category_list(obs.obs_properties_get(props, "category"), game)
    return True


# --- Autosplitting (image matching) ------------------------------------------

def load_template(path):
    """Reads an image file as (bgr, mask).

    `bgr` is the cv2 BGR pixel data. `mask` is the source PNG's alpha
    channel (255 = opaque, 0 = fully transparent) if it has one and it
    isn't uniformly opaque, else None to match the whole image.

    cv2.imread() silently fails on Windows for paths containing
    non-ASCII characters (e.g. accented game names), so the file is read
    as bytes and decoded instead.
    """
    import cv2
    import numpy as np

    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    if image.ndim == 2:  # grayscale, no alpha
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), None
    if image.shape[2] == 4:
        alpha = image[:, :, 3]
        mask = None if alpha.min() == 255 else alpha
        return image[:, :, :3], mask
    return image, None


def crop_roi(frame, roi):
    """Crops `frame` to `roi` = [x, y, w, h], clipped to the frame bounds.

    Returns None if the clipped rect is empty (roi fully outside the frame).
    """
    fh, fw = frame.shape[:2]
    x, y, w, h = roi
    x0, y0 = max(0, min(x, fw)), max(0, min(y, fh))
    x1, y1 = max(0, min(x + w, fw)), max(0, min(y + h, fh))
    if x1 <= x0 or y1 <= y0:
        return None
    return frame[y0:y1, x0:x1]


def prepare_masked_template(bgr, mask):
    """Precomputes the parts of a masked normalized cross-correlation that
    only depend on the template, not the frame it's matched against later.

    cv2.matchTemplate has no TM_CCOEFF_NORMED support for masked templates
    (only TM_SQDIFF/TM_CCORR_NORMED), and TM_CCORR_NORMED alone has no mean
    subtraction, so unlike TM_CCOEFF_NORMED its score for a genuine
    non-match doesn't drop much below ~0.7-0.8. This reimplements
    TM_CCOEFF_NORMED's formula restricted to the unmasked pixels, expanded
    into a handful of unmasked cv2.matchTemplate(TM_CCORR) calls (see
    masked_match_score) so the per-frame work still runs through cv2's own
    optimized correlation rather than a Python pixel/window loop.
    """
    import numpy as np

    m = mask.astype(np.float32) / 255.0
    sum_m = float(m.sum())
    bgr_f = bgr.astype(np.float32)
    t_prime = np.empty_like(bgr_f)
    for c in range(3):
        mean_c = float((m * bgr_f[:, :, c]).sum() / sum_m)
        t_prime[:, :, c] = m * (bgr_f[:, :, c] - mean_c)
    denom_t = float((t_prime ** 2).sum())
    return t_prime, m, sum_m, denom_t


def masked_match_score(frame, masked_template):
    """Best score of a masked_template (from prepare_masked_template)
    against every position it fits in `frame`."""
    import cv2
    import numpy as np

    t_prime, m, sum_m, denom_t = masked_template
    frame_f = frame.astype(np.float32)
    numerator = cv2.matchTemplate(frame_f, t_prime, cv2.TM_CCORR)

    denom_i = np.zeros(numerator.shape, dtype=np.float32)
    for c in range(3):
        channel = frame_f[:, :, c]
        s1 = cv2.matchTemplate(channel, m, cv2.TM_CCORR)
        s2 = cv2.matchTemplate(channel ** 2, m, cv2.TM_CCORR)
        denom_i += s2 - (s1 ** 2) / sum_m

    denom = np.sqrt(np.maximum(denom_t * denom_i, 0)) + 1e-8
    return float((numerator / denom).max())


def flat_match_score(frame, template):
    """Match score for a near-uniform-color `template` (see
    FLAT_TEMPLATE_STD) against `frame`: 1 minus the single worst-matching
    pixel's max-channel absolute difference from the template's own
    (uniform) color, as a fraction of the 0-255 range.

    Unlike match_score's other two branches, this does not slide the
    template across `frame` looking for the best-fitting position - every
    splits.json entry pairs a flat template with a same-size roi, so
    `frame` (already cropped to that roi by match_score) is the single
    region being judged.
    """
    import numpy as np

    color = template.reshape(-1, template.shape[2]).mean(axis=0)
    worst_pixel_diff = float(np.abs(frame.astype(np.float32) - color).max())
    return 1.0 - worst_pixel_diff / 255


def prepare_template(bgr, mask):
    """Wraps a loaded (bgr, mask) pair (see load_template) into whatever
    match_score needs, precomputing the masked case's template-only work
    once at load time instead of on every matched frame.

    The second element of the returned tuple is the masked_match_score
    data if `mask` was given, the string "flat" if the template is a
    near-uniform color (see FLAT_TEMPLATE_STD), or else None.
    """
    if mask is not None:
        return None, prepare_masked_template(bgr, mask)
    if bgr.std() < FLAT_TEMPLATE_STD:
        return bgr, "flat"
    return bgr, None


def match_score(frame, template, roi=None):
    """Best matchTemplate score of `template` (as returned by
    prepare_template) in `frame` (or within `roi` if given), or None if
    the search area is too small for the template."""
    import cv2

    bgr, extra = template
    if roi is not None:
        frame = crop_roi(frame, roi)
        if frame is None:
            return None

    th, tw = (extra[0].shape[:2] if bgr is None else bgr.shape[:2])
    fh, fw = frame.shape[:2]
    if th > fh or tw > fw:
        return None

    if bgr is None:
        return masked_match_score(frame, extra)
    if extra == "flat":
        return flat_match_score(frame, bgr)
    return float(cv2.matchTemplate(frame, bgr, cv2.TM_CCOEFF_NORMED).max())


def validate_roi(label, roi):
    if (not isinstance(roi, (list, tuple)) or len(roi) != 4
            or not all(isinstance(n, int) for n in roi)):
        raise ValueError(
            f"'{label}' has an invalid roi (expected [x, y, w, h] ints): {roi!r}")


def parse_image_entry(key, entry):
    """One "images" list entry for split `key`: either a filename (matched
    against the full frame) or {filename: [x, y, w, h]} restricting
    matchTemplate to that rect just for this image.

    Returns (filename, roi_or_None).
    """
    if isinstance(entry, str):
        return entry, None
    if isinstance(entry, dict) and len(entry) == 1:
        (name, roi), = entry.items()
        validate_roi(f"{key}: {name}", roi)
        return name, roi
    raise ValueError(f"'{key}' has an invalid images entry: {entry!r}")


def ordered_split_keys(splits):
    """['start', then numeric keys ascending, then 'stop']."""
    def sort_key(key):
        if key == "start":
            return (0, 0)
        if key == "stop":
            return (2, 0)
        return (1, int(key))
    return sorted(splits.keys(), key=sort_key)


def load_splits(category_dir):
    path = os.path.join(category_dir, SPLITS_FILENAME)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def standalone_connect_livesplit(url):
    """Connects to the LiveSplit WS server, retrying until it succeeds."""
    import websocket

    failure_logged = False
    while True:
        try:
            return websocket.create_connection(
                url, timeout=LIVESPLIT_CONNECT_TIMEOUT_S)
        except Exception as e:
            if not failure_logged:
                log("WARNING",
                    f"Cannot reach LiveSplit at {url} ({e}). Make sure the "
                    "WS server is running: right click LiveSplit -> Control "
                    f"-> Start WS Server. Retrying every {LIVESPLIT_RETRY_S}s...")
                failure_logged = True
            time.sleep(LIVESPLIT_RETRY_S)


def send_livesplit_command(url, command):
    """Opens a fresh connection, sends `command`, and closes it.

    Splits in a single run can be many minutes apart, and a WebSocket left
    open and idle for that long can go stale without either side noticing
    (send() on a half-dead connection can return successfully even though
    the server never receives it, since nothing here calls recv() to
    detect a server-side close). A short-lived connection per command
    sidesteps that entirely.
    """
    conn = standalone_connect_livesplit(url)
    try:
        conn.send(command)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def livesplit_reset_watcher(url, reset_event, stop):
    """Sets `reset_event` whenever LiveSplit's timer goes from Running/Paused
    back to NotRunning, i.e. it was reset by something other than this
    script (a hotkey, the LiveSplit UI, another tool).

    The LiveSplit server only ever answers commands it is sent; it never
    pushes phase changes on its own, so polling 'getcurrenttimerphase' is
    the only way to notice such a reset. Runs on its own short-lived
    connections, like send_livesplit_command, so a slow/unreachable server
    can never stall the frame-matching loop this runs alongside.
    """
    import websocket

    was_active = False
    while not stop.is_set():
        try:
            conn = websocket.create_connection(
                url, timeout=LIVESPLIT_CONNECT_TIMEOUT_S)
            try:
                conn.settimeout(LIVESPLIT_CONNECT_TIMEOUT_S)
                conn.send("getcurrenttimerphase")
                phase = conn.recv().strip()
            finally:
                conn.close()
        except Exception:
            stop.wait(RESET_POLL_INTERVAL_S)
            continue

        if phase in ("Running", "Paused"):
            was_active = True
        elif phase == "NotRunning" and was_active:
            was_active = False
            reset_event.set()
        stop.wait(RESET_POLL_INTERVAL_S)


def disable_process_power_throttling():
    """Opts this process out of Windows' Efficiency Mode / EcoQoS power
    throttling.

    Windows can silently throttle CPU scheduling for background processes
    with no visible window and no recent input — exactly how the
    --autosplit subprocess runs (spawned with CREATE_NO_WINDOW) — which
    can make camera reads lag behind the live feed by whole seconds even
    though nothing in this file's own code is slow.
    """
    class PROCESS_POWER_THROTTLING_STATE(ctypes.Structure):
        _fields_ = [
            ("Version", ctypes.c_uint32),
            ("ControlMask", ctypes.c_uint32),
            ("StateMask", ctypes.c_uint32),
        ]

    PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1
    ProcessPowerThrottling = 4  # PROCESS_INFORMATION_CLASS enum value

    state = PROCESS_POWER_THROTTLING_STATE(
        Version=1,
        ControlMask=PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
        StateMask=0)  # 0 = do not throttle
    try:
        ctypes.windll.kernel32.SetProcessInformation(
            ctypes.windll.kernel32.GetCurrentProcess(),
            ProcessPowerThrottling,
            ctypes.byref(state), ctypes.sizeof(state))
    except (OSError, AttributeError):
        pass  # best-effort; not available on older Windows versions


class LatestFrame:
    """Holds only the most recently grabbed camera frame.

    Written by frame_grabber_loop (on its own thread) and read by the
    matching loop, so matching always sees the newest frame instead of
    one still sitting behind a backlog in cv2's own internal buffering.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._ok = False
        self._frame = None

    def set(self, ok, frame):
        with self._lock:
            self._ok = ok
            self._frame = frame

    def get(self):
        with self._lock:
            return self._ok, self._frame


def frame_grabber_loop(cap, latest, stop):
    """Reads the camera as fast as it supplies frames, discarding any
    backlog by only ever keeping the newest one for the matching loop."""
    while not stop.is_set():
        ok, frame = cap.read()
        latest.set(ok, frame)
        if not ok:
            time.sleep(FRAME_POLL_INTERVAL_S)


def standalone_autosplit_main(game, category, livesplit_url):
    """Matches the camera feed against splits.json and drives LiveSplit.

    Runs as its own OS process (spawned by restart_autosplit_process),
    since cv2.VideoCapture(CAP_DSHOW) only gets real frames from the OBS
    Virtual Camera when opened outside OBS's own process.

    Waits for the category's local files (which may still be downloading
    from GitHub) to become available, then walks the splits in order
    ('start', numeric keys ascending, 'stop'). Within a split, its images
    must match in the given order before the split's delay starts; only
    then is the LiveSplit command sent and the next split becomes active.
    'start' sends 'starttimer'; every other split (including 'stop', which
    ends the run) sends 'split'.

    Rearms back to 'start' both after 'stop' is reached and whenever
    LiveSplit's timer is reset by anything other than this script (see
    livesplit_reset_watcher), so repeated attempts don't need OBS or the
    subprocess restarted between them.
    """
    category_dir = os.path.join(GAMES_DIR, game, category)
    splits_path = os.path.join(category_dir, SPLITS_FILENAME)

    while not os.path.isfile(splits_path):
        time.sleep(AUTOSPLIT_WAIT_S)

    try:
        splits = load_splits(category_dir)
        keys = ordered_split_keys(splits)
        templates = {}
        rois = {}
        for key in keys:
            parsed = [parse_image_entry(key, entry) for entry in splits[key]["images"]]
            templates[key] = [prepare_template(*load_template(os.path.join(category_dir, "images", name)))
                              for name, _ in parsed]
            rois[key] = [roi for _, roi in parsed]
    except (OSError, ValueError, KeyError) as e:
        log("ERROR", f"Could not load splits for '{game}/{category}': {e}")
        return

    camera_index = find_index_by_name()
    if camera_index < 0:
        log("ERROR", "Could not find the OBS Virtual Camera index")
        return

    import cv2

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        log("ERROR", "Could not open the camera for template matching")
        return
    # Without this, cv2 can queue up several frames internally and read()
    # keeps returning the oldest one once the queue is full, making
    # matches (and therefore splits) lag behind the live feed.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Verify LiveSplit is reachable before arming (fails fast with a clear
    # log line if it's not); each actual split is then sent over its own
    # short-lived connection (see send_livesplit_command).
    standalone_connect_livesplit(livesplit_url).close()
    log("INFO", f"Connected to LiveSplit at {livesplit_url}")
    log("INFO", f"Autosplitter armed for '{game}/{category}': {' -> '.join(keys)}")

    latest_frame = LatestFrame()
    stop_grabbing = threading.Event()
    grabber_thread = threading.Thread(
        target=frame_grabber_loop, args=(cap, latest_frame, stop_grabbing),
        daemon=True)
    grabber_thread.start()

    reset_event = threading.Event()
    stop_watching_reset = threading.Event()
    reset_watcher_thread = threading.Thread(
        target=livesplit_reset_watcher,
        args=(livesplit_url, reset_event, stop_watching_reset), daemon=True)
    reset_watcher_thread.start()

    key_index = 0
    image_index = 0
    read_failure_logged = False
    last_score_log = 0.0
    try:
        while True:
            if key_index >= len(keys):
                log("INFO",
                    f"Autosplitter finished '{game}/{category}'; waiting "
                    "for the next attempt")
                key_index = 0
                image_index = 0

            if reset_event.is_set():
                reset_event.clear()
                if key_index > 0:
                    log("INFO",
                        "LiveSplit was reset; rearming autosplitter for "
                        "the next attempt")
                key_index = 0
                image_index = 0

            ok, frame = latest_frame.get()
            if not ok or frame is None:
                if not read_failure_logged:
                    log("WARNING",
                        "Camera read failed (no frames from the virtual "
                        "camera); will keep retrying")
                    read_failure_logged = True
                time.sleep(FRAME_POLL_INTERVAL_S)
                continue
            read_failure_logged = False

            key = keys[key_index]
            score = match_score(frame, templates[key][image_index], rois[key][image_index])
            if score is None or score < MATCH_THRESHOLD:
                now = time.monotonic()
                if now - last_score_log >= SCORE_LOG_INTERVAL_S:
                    name, _ = parse_image_entry(key, splits[key]["images"][image_index])
                    shown = "n/a (roi outside frame)" if score is None else f"{score:.3f}"
                    log("INFO",
                        f"Waiting on '{key}' image '{name}', "
                        f"score={shown} (need {MATCH_THRESHOLD})")
                    last_score_log = now
                time.sleep(FRAME_POLL_INTERVAL_S)
                continue

            image_index += 1
            if image_index < len(templates[key]):
                continue  # this split's next image may already be on screen

            delay_s = splits[key].get("delay", 0) / 1000
            log("INFO", f"'{key}' detected, acting in {delay_s:.1f}s")
            time.sleep(delay_s)

            command = "starttimer" if key == "start" else "split"
            try:
                send_livesplit_command(livesplit_url, command)
            except Exception as e:
                log("ERROR", f"Could not send '{command}' to LiveSplit: {e}")

            key_index += 1
            image_index = 0
    finally:
        stop_watching_reset.set()
        reset_watcher_thread.join(timeout=2)
        stop_grabbing.set()
        grabber_thread.join(timeout=2)
        cap.release()


def read_autosplit_output(process):
    """Relays the --autosplit subprocess's stdout into the OBS script log,
    restoring the level each line was printed with (see log())."""
    for line in process.stdout:
        line = line.rstrip("\n")
        if not line:
            continue
        level, sep, msg = line.partition(": ")
        if sep and level == "ERROR":
            log("ERROR", msg)
        elif sep and level == "WARNING":
            log("WARNING", msg)
        elif sep and level == "INFO":
            log("INFO", msg)
        else:
            log("INFO", line)


def stop_autosplit_process():
    global autosplit_process
    if autosplit_process is None:
        return
    try:
        autosplit_process.terminate()
        autosplit_process.wait(timeout=5)
    except Exception:
        try:
            autosplit_process.kill()
        except Exception:
            pass
    autosplit_process = None


def restart_autosplit_process(game, category):
    """Spawns the --autosplit subprocess for `game`/`category`, stopping any
    prior one first."""
    global autosplit_process, autosplit_reader_thread

    stop_autosplit_process()

    if not game or not category:
        autosplit_reader_thread = None
        return

    python_exe = os.path.join(sys.base_prefix, "python.exe")
    if not os.path.isfile(python_exe):
        log("ERROR",
            f"Cannot start autosplitter: python.exe not found in {sys.base_prefix}")
        return

    autosplit_process = subprocess.Popen(
        [python_exe, SCRIPT_PATH, "--autosplit", game, category, livesplit_url],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8",
        # ABOVE_NORMAL, plus disable_process_power_throttling() on the
        # subprocess side: a windowless background process is prone to
        # Windows deprioritizing its scheduling, which showed up as
        # camera reads lagging seconds behind the live feed.
        creationflags=subprocess.CREATE_NO_WINDOW
                     | subprocess.ABOVE_NORMAL_PRIORITY_CLASS)
    autosplit_reader_thread = threading.Thread(
        target=read_autosplit_output, args=(autosplit_process,), daemon=True)
    autosplit_reader_thread.start()


# --- OBS script entry points -------------------------------------------------

def script_description():
    return ("<b>Autosplitter</b><br/>"
            "Starts the OBS Virtual Camera on load and detects its cv2 "
            "camera index.<br/>"
            "Connects to the LiveSplit WS server (start it via right click "
            "on LiveSplit &rarr; Control &rarr; Start WS Server).<br/>"
            "Missing dependencies (<code>opencv-python</code>, "
            "<code>pygrabber</code>, <code>websocket-client</code>) are "
            "installed automatically on first run; see the script log for "
            "progress.")


def script_defaults(settings):
    obs.obs_data_set_default_string(settings, "livesplit_host", "localhost")
    obs.obs_data_set_default_int(settings, "livesplit_port", 16834)


def script_update(settings):
    global livesplit_url, current_settings, autosplit_game, autosplit_category
    current_settings = settings

    host = obs.obs_data_get_string(settings, "livesplit_host").strip() or "localhost"
    if host.lower() == "localhost":
        # On Windows, connecting to "localhost" tries IPv6 (::1) first and
        # only falls back to IPv4 after a ~2s timeout when nothing answers
        # there, since LiveSplit's WS server only listens on IPv4. That 2s
        # was landing on every single split once each one got its own
        # short-lived connection (see send_livesplit_command).
        host = "127.0.0.1"
    port = obs.obs_data_get_int(settings, "livesplit_port")
    url = f"ws://{host}:{port or 16834}/livesplit"
    url_changed = url != livesplit_url
    if url_changed:
        livesplit_url = url
        restart_livesplit_worker()

    game = obs.obs_data_get_string(settings, "game")
    category = obs.obs_data_get_string(settings, "category")
    category_changed = (game, category) != (autosplit_game, autosplit_category)
    if category_changed:
        autosplit_game, autosplit_category = game, category
    # A LiveSplit URL change also needs to reach an already-armed
    # subprocess, which connected to LiveSplit itself at spawn time.
    if category_changed or url_changed:
        restart_autosplit_process(game, category)


def script_load(settings):
    global camera_index, start_attempts, unloading, current_settings
    camera_index = -1
    start_attempts = 0
    unloading = False
    current_settings = settings
    # Delay startup via a timer: at OBS launch the frontend may not be
    # ready yet, and the timer also retries until the camera is active.
    obs.timer_add(startup_tick, RETRY_INTERVAL_MS)


def script_unload():
    global unloading
    unloading = True
    obs.timer_remove(startup_tick)
    stop_autosplit_process()
    if livesplit_stop is not None:
        livesplit_stop.set()
    close_livesplit_conn()
    if started_by_script and obs.obs_frontend_virtualcam_active():
        obs.obs_frontend_stop_virtualcam()


def script_properties():
    props = obs.obs_properties_create()

    game_prop = obs.obs_properties_add_list(
        props, "game", "Game",
        obs.OBS_COMBO_TYPE_LIST, obs.OBS_COMBO_FORMAT_STRING)
    category_prop = obs.obs_properties_add_list(
        props, "category", "Category",
        obs.OBS_COMBO_TYPE_LIST, obs.OBS_COMBO_FORMAT_STRING)
    fill_game_list(game_prop)
    selected_game = (obs.obs_data_get_string(current_settings, "game")
                      if current_settings else "")
    fill_category_list(category_prop, selected_game)
    obs.obs_property_set_modified_callback(game_prop, game_changed)
    obs.obs_property_set_modified_callback(category_prop, category_changed)

    obs.obs_properties_add_button(props, "redetect",
                                  "Re-detect camera index", redetect_clicked)
    obs.obs_properties_add_text(props, "livesplit_host", "LiveSplit host",
                                obs.OBS_TEXT_DEFAULT)
    obs.obs_properties_add_int(props, "livesplit_port", "LiveSplit port",
                               1, 65535, 1)
    obs.obs_properties_add_button(props, "livesplit_test",
                                  "Test LiveSplit connection",
                                  test_livesplit_clicked)
    return props


# --- Standalone entry point --------------------------------------------------
# Reached only when OBS's autosplit process runs this same file as
# `python.exe script.py --autosplit <game> <category> <livesplit_url>`;
# never when OBS imports it as a script module.
if __name__ == "__main__":
    # Redirected to a pipe, stdout/stderr would otherwise default to the
    # system codepage (e.g. cp1252) instead of UTF-8, which read_autosplit_
    # output() in the parent process expects (game/category names may
    # contain non-ASCII characters, e.g. accented ones).
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    disable_process_power_throttling()
    if len(sys.argv) == 5 and sys.argv[1] == "--autosplit":
        standalone_autosplit_main(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("usage: script.py --autosplit <game> <category> <livesplit_url>",
              file=sys.stderr)
        sys.exit(1)
