"""OBS autosplitter script.

On load, starts the OBS Virtual Camera and determines which cv2 (OpenCV)
device index it is reachable under, so frames can later be captured with
cv2.VideoCapture(camera_index, cv2.CAP_DSHOW).

Also keeps a WebSocket connection to the LiveSplit server open
(ws://<host>:<port>/livesplit, default port 16834) and reconnects
automatically. The WS server must be started inside LiveSplit first:
right click LiveSplit -> Control -> Start WS Server. Timer commands
(starttimer, split, ...) are sent with livesplit_send().

Dependencies (opencv-python, pygrabber, websocket-client) are installed
automatically into the user site-packages of OBS's configured Python
when missing. Manual install, if ever needed:
    C:\Python311\python.exe -m pip install --user opencv-python pygrabber websocket-client
pygrabber finds the camera by its DirectShow device name; without it the
script falls back to probing every index.
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

import obspython as obs

RETRY_INTERVAL_MS = 1000
MAX_START_ATTEMPTS = 15
PROBE_MAX_INDEX = 10
PIP_TIMEOUT_S = 600
LIVESPLIT_RETRY_S = 5
LIVESPLIT_CONNECT_TIMEOUT_S = 5
# recv() timeout; also how quickly the worker notices a stop request.
LIVESPLIT_RECV_TIMEOUT_S = 1

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GAMES_DIR = os.path.join(SCRIPT_DIR, "games")
SPLITS_FILENAME = "splits.json"

# cv2.matchTemplate (TM_CCOEFF_NORMED) score above which a template counts
# as detected in the current frame; tune per game if splits misfire.
MATCH_THRESHOLD = 0.9
# An "images" list entry in splits.json may be either a filename (matched
# against the full frame) or {filename: [x, y, w, h]} (pixels, in the
# camera frame's own coordinate space) to restrict matchTemplate to that
# rect for just that image. Narrows the search space (faster) and ignores
# unrelated on-screen changes (fewer false positives).
# How often the autosplitter re-reads the camera while a split's images
# haven't matched yet.
FRAME_POLL_INTERVAL_S = 0.1
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

# game/category currently armed in the autosplit worker (script_update
# restarts the worker when these no longer match the settings).
autosplit_game = ""
autosplit_category = ""
autosplit_thread = None
autosplit_stop = None

# Settings from the last script_load()/script_update(), kept so
# script_properties() can populate the category list for the game that is
# already selected when the properties dialog is (re)opened.
current_settings = None


def get_camera_index():
    """cv2 device index of the OBS Virtual Camera, or -1 if not found."""
    return camera_index


def log(level, msg):
    if not unloading:
        obs.script_log(level, msg)


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
            log(obs.LOG_ERROR,
                f"Cannot auto-install {missing}: python.exe not found in "
                f"{sys.base_prefix}. Install manually with: "
                f"pip install --user {' '.join(missing)}")
            return False

        log(obs.LOG_INFO,
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
            log(obs.LOG_ERROR, f"Could not run pip install: {e}")
            return False
        if result.returncode != 0:
            tail = (result.stderr or result.stdout
                    or "").strip().splitlines()[-5:]
            log(obs.LOG_ERROR, "pip install failed:\n" + "\n".join(tail))
            return False

        # A fresh --user install may land in a directory that was not on
        # sys.path when the interpreter started.
        user_site = site.getusersitepackages()
        if user_site not in sys.path:
            sys.path.insert(0, user_site)
        importlib.invalidate_caches()
        log(obs.LOG_INFO, f"Installed {', '.join(missing)}")
        return True


def find_index_by_name():
    # pygrabber enumerates DirectShow devices in the same order cv2 uses
    # with the CAP_DSHOW backend, so the list position is the cv2 index.
    try:
        from pygrabber.dshow_graph import FilterGraph

        devices = FilterGraph().get_input_devices()
    except ImportError:
        log(obs.LOG_WARNING,
            "pygrabber not installed, falling back to probing indices "
            "by resolution (pip install pygrabber)")
        return -1
    except OSError as e:
        log(obs.LOG_WARNING,
            f"pygrabber device enumeration failed ({e}), falling back "
            "to probing indices by resolution")
        return -1

    for i, name in enumerate(devices):
        if "OBS Virtual Camera" in name:
            log(obs.LOG_INFO,
                f"Found '{name}' at cv2 index {i} (by device name)")
            return i
    log(obs.LOG_WARNING, f"OBS Virtual Camera not in device list: {devices}")
    return -1


def find_index_by_resolution():
    # Fallback: open each index and compare its resolution to the OBS
    # output resolution, which is what the virtual camera outputs.
    try:
        import cv2
    except ImportError:
        log(obs.LOG_ERROR,
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
            log(obs.LOG_INFO,
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
        log(obs.LOG_ERROR, "Could not find the OBS Virtual Camera index")


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
            log(obs.LOG_ERROR,
                "Giving up: virtual camera did not start after "
                f"{MAX_START_ATTEMPTS} attempts")
            return
        obs.obs_frontend_start_virtualcam()
        started_by_script = True
        return  # verify it is active on the next tick

    obs.timer_remove(startup_tick)
    log(obs.LOG_INFO, "Virtual camera is active")
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
        log(obs.LOG_WARNING,
            f"Not connected to LiveSplit, dropped command '{command}'")
        return False
    try:
        conn.send(command)
        return True
    except Exception as e:
        log(obs.LOG_WARNING, f"Sending '{command}' to LiveSplit failed: {e}")
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
        log(obs.LOG_ERROR,
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
                log(obs.LOG_WARNING,
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
        log(obs.LOG_INFO, f"Connected to LiveSplit at {url}")

        try:
            while not stop.is_set():
                try:
                    msg = conn.recv()
                except websocket.WebSocketTimeoutException:
                    continue
                if msg:
                    log(obs.LOG_INFO, f"LiveSplit: {msg}")
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
            log(obs.LOG_WARNING, "LiveSplit connection lost, reconnecting...")


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
        log(obs.LOG_INFO,
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
        log(obs.LOG_WARNING, f"Could not list GitHub path '{path}' ({e})")
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
        log(obs.LOG_WARNING, f"Could not download '{remote_path}' from GitHub ({e})")
        return False
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        with open(local_path, "wb") as f:
            f.write(data)
    except OSError as e:
        log(obs.LOG_WARNING, f"Could not save '{local_path}' ({e})")
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
        log(obs.LOG_INFO, f"Downloaded '{game}/{category}' from GitHub")
    else:
        log(obs.LOG_WARNING,
            f"'{game}/{category}' download from GitHub finished with errors")


def download_category_if_missing(game, category):
    if not game or not category or category_exists_locally(game, category):
        return
    log(obs.LOG_INFO, f"'{game}/{category}' not found locally, downloading from GitHub...")
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
    """Reads an image file as a cv2 (BGR) array.

    cv2.imread() silently fails on Windows for paths containing
    non-ASCII characters (e.g. accented game names), so the file is read
    as bytes and decoded instead.
    """
    import cv2
    import numpy as np

    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


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


def frame_matches(frame, template, roi=None):
    import cv2

    if roi is not None:
        frame = crop_roi(frame, roi)
        if frame is None:
            return False
    th, tw = template.shape[:2]
    fh, fw = frame.shape[:2]
    if th > fh or tw > fw:
        return False
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    return bool(result.max() >= MATCH_THRESHOLD)


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


def autosplit_worker(game, category, stop):
    """Matches the camera feed against splits.json and drives LiveSplit.

    Waits for the camera index and the category's local files (which may
    still be downloading from GitHub) to become available, then walks the
    splits in order ('start', numeric keys ascending, 'stop'). Within a
    split, its images must match in the given order before the split's
    delay starts; only then is the LiveSplit command sent and the next
    split becomes active. 'start' sends 'starttimer'; every other split
    (including 'stop', which ends the run) sends 'split'.
    """
    category_dir = os.path.join(GAMES_DIR, game, category)
    splits_path = os.path.join(category_dir, SPLITS_FILENAME)

    while not stop.is_set() and (camera_index < 0 or not os.path.isfile(splits_path)):
        stop.wait(AUTOSPLIT_WAIT_S)
    if stop.is_set():
        return

    try:
        splits = load_splits(category_dir)
        keys = ordered_split_keys(splits)
        templates = {}
        rois = {}
        for key in keys:
            parsed = [parse_image_entry(key, entry) for entry in splits[key]["images"]]
            templates[key] = [load_template(os.path.join(category_dir, "images", name))
                              for name, _ in parsed]
            rois[key] = [roi for _, roi in parsed]
    except (OSError, ValueError, KeyError) as e:
        log(obs.LOG_ERROR, f"Could not load splits for '{game}/{category}': {e}")
        return

    import cv2

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        log(obs.LOG_ERROR, "Autosplitter: could not open the camera for template matching")
        return

    log(obs.LOG_INFO, f"Autosplitter armed for '{game}/{category}': {' -> '.join(keys)}")
    key_index = 0
    image_index = 0
    try:
        while not stop.is_set() and key_index < len(keys):
            ok, frame = cap.read()
            if not ok:
                stop.wait(FRAME_POLL_INTERVAL_S)
                continue

            key = keys[key_index]
            if not frame_matches(frame, templates[key][image_index], rois[key][image_index]):
                stop.wait(FRAME_POLL_INTERVAL_S)
                continue

            image_index += 1
            if image_index < len(templates[key]):
                continue  # this split's next image may already be on screen

            delay_s = splits[key].get("delay", 0) / 1000
            log(obs.LOG_INFO, f"Autosplitter: '{key}' detected, "
                              f"acting in {delay_s:.1f}s")
            stop.wait(delay_s)
            if stop.is_set():
                break
            livesplit_send("starttimer" if key == "start" else "split")

            key_index += 1
            image_index = 0
    finally:
        cap.release()

    if key_index >= len(keys):
        log(obs.LOG_INFO, f"Autosplitter finished '{game}/{category}'")


def restart_autosplit_worker(game, category):
    """Starts the autosplit worker for `game`/`category`, stopping any prior one."""
    global autosplit_thread, autosplit_stop

    if autosplit_stop is not None:
        autosplit_stop.set()

    if not game or not category:
        autosplit_thread = None
        autosplit_stop = None
        return

    autosplit_stop = threading.Event()
    autosplit_thread = threading.Thread(
        target=autosplit_worker, args=(game, category, autosplit_stop), daemon=True)
    autosplit_thread.start()


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

    host = obs.obs_data_get_string(settings, "livesplit_host").strip()
    port = obs.obs_data_get_int(settings, "livesplit_port")
    url = f"ws://{host or 'localhost'}:{port or 16834}/livesplit"
    if url != livesplit_url:
        livesplit_url = url
        restart_livesplit_worker()

    game = obs.obs_data_get_string(settings, "game")
    category = obs.obs_data_get_string(settings, "category")
    if (game, category) != (autosplit_game, autosplit_category):
        autosplit_game, autosplit_category = game, category
        restart_autosplit_worker(game, category)


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
    if autosplit_stop is not None:
        autosplit_stop.set()
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
