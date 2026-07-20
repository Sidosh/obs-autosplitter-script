"""OBS autosplitter script.

On load, starts the OBS Virtual Camera and determines which cv2 (OpenCV)
device index it is reachable under, so frames can later be captured with
cv2.VideoCapture(camera_index, cv2.CAP_DSHOW).

Dependencies (opencv-python, pygrabber) are installed automatically into
the user site-packages of OBS's configured Python when missing. Manual
install, if ever needed:
    C:\Python311\python.exe -m pip install --user opencv-python pygrabber
pygrabber finds the camera by its DirectShow device name; without it the
script falls back to probing every index.
"""

import ctypes
import importlib
import importlib.util
import os
import site
import subprocess
import sys
import threading

import obspython as obs

RETRY_INTERVAL_MS = 1000
MAX_START_ATTEMPTS = 15
PROBE_MAX_INDEX = 10
PIP_TIMEOUT_S = 600

# import name -> pip package name
REQUIRED_PACKAGES = {
    "cv2": "opencv-python",
    "pygrabber": "pygrabber",
}

camera_index = -1
start_attempts = 0
started_by_script = False
unloading = False
detect_thread = None


def get_camera_index():
    """cv2 device index of the OBS Virtual Camera, or -1 if not found."""
    return camera_index


def log(level, msg):
    if not unloading:
        obs.script_log(level, msg)


def ensure_dependencies():
    """Install missing packages via pip into the user site-packages.

    Runs on the worker thread, so OBS stays responsive during the
    download/install.
    """
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
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
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


# --- OBS script entry points -------------------------------------------------

def script_description():
    return ("<b>Autosplitter</b><br/>"
            "Starts the OBS Virtual Camera on load and detects its cv2 "
            "camera index.<br/>"
            "Missing dependencies (<code>opencv-python</code>, "
            "<code>pygrabber</code>) are installed automatically on first "
            "run; see the script log for progress.")


def script_load(settings):
    global camera_index, start_attempts, unloading
    camera_index = -1
    start_attempts = 0
    unloading = False
    # Delay startup via a timer: at OBS launch the frontend may not be
    # ready yet, and the timer also retries until the camera is active.
    obs.timer_add(startup_tick, RETRY_INTERVAL_MS)


def script_unload():
    global unloading
    unloading = True
    obs.timer_remove(startup_tick)
    if started_by_script and obs.obs_frontend_virtualcam_active():
        obs.obs_frontend_stop_virtualcam()


def script_properties():
    props = obs.obs_properties_create()
    obs.obs_properties_add_button(props, "redetect",
                                  "Re-detect camera index", redetect_clicked)
    return props
