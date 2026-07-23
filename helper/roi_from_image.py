"""Standalone helper: find where a picture appears in the OBS Virtual
Camera output right now, and print its roi.

Start the virtual camera in OBS first (or load script.py, which starts
it), then run this with the same Python OBS uses so opencv-python is
available:
    C:\\Python311\\python.exe helper\\roi_from_image.py <path-to-image>

The path does not need quotes even if it contains spaces (e.g. game/category
folder names like "Pokémon Red-Blue"): PowerShell/cmd split an unquoted path
into several argv entries at each space, so any extra argv entries are
rejoined with spaces to recover the original path.

The picture is located in the current camera frame with cv2.matchTemplate
(the same routine script.py's own match_score uses), and the best-matching
position is printed as a ready-to-paste splits.json images entry, together
with the match score so you can tell whether it actually found the picture
on screen or just picked the least-bad location:
    {"<filename>": [x, y, w, h]}  (score=...)
"""

import os
import sys

# script.py lives at the repo root, one level up from this helper.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import script

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is not installed for this interpreter. "
             "Install it with:\n"
             f"    {sys.executable} -m pip install --user opencv-python")

# Frames to read before capturing, so the feed has settled after opening.
WARMUP_FRAMES = 5


def find_camera_index():
    """cv2 device index of the OBS Virtual Camera, or -1 if not found.

    pygrabber enumerates DirectShow devices in the same order cv2 uses
    with the CAP_DSHOW backend, so the list position is the cv2 index.
    """
    sys.coinit_flags = 2  # COINIT_APARTMENTTHREADED; comtypes reads this
    try:
        from pygrabber.dshow_graph import FilterGraph

        devices = FilterGraph().get_input_devices()
    except (ImportError, OSError) as e:
        print(f"Cannot enumerate camera devices ({e}); "
              "pass the index with --index", file=sys.stderr)
        return -1

    for i, name in enumerate(devices):
        if "OBS Virtual Camera" in name:
            print(f"Found '{name}' at cv2 index {i}")
            return i
    print(f"OBS Virtual Camera not in device list: {devices}\n"
          "Is the virtual camera started in OBS?", file=sys.stderr)
    return -1


def grab_frame(cap):
    """Read one frame after a few warm-up reads, or None on failure."""
    frame = None
    for _ in range(WARMUP_FRAMES):
        ok, frame = cap.read()
        if not ok:
            return None
    return frame


def locate(frame, bgr):
    """Best matchTemplate position of `bgr` within `frame`.

    Returns (x, y, w, h, score); (w, h) is `bgr`'s own size, since that is
    the size of the window matchTemplate slides across `frame`.
    """
    result = cv2.matchTemplate(frame, bgr, cv2.TM_CCOEFF_NORMED)
    _, score, _, (x, y) = cv2.minMaxLoc(result)
    h, w = bgr.shape[:2]
    return x, y, w, h, score


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: roi_from_image.py <path-to-image>")
    path = " ".join(sys.argv[1:])

    try:
        bgr, _ = script.load_template(path)
    except (OSError, ValueError) as e:
        sys.exit(f"Could not read image: {e}")

    index = find_camera_index()
    if index < 0:
        sys.exit(1)

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(f"Could not open camera index {index}")
    try:
        frame = grab_frame(cap)
    finally:
        cap.release()
    if frame is None:
        sys.exit("Could not read a frame (virtual camera stopped?)")

    fh, fw = frame.shape[:2]
    th, tw = bgr.shape[:2]
    if th > fh or tw > fw:
        sys.exit(f"Image ({tw}x{th}) is larger than the camera frame ({fw}x{fh})")

    x, y, w, h, score = locate(frame, bgr)
    name = os.path.basename(path)
    print(f'{{"{name}": [{x}, {y}, {w}, {h}]}}  (score={score:.3f})')
    if score < script.MATCH_THRESHOLD:
        print(f"Note: score is below script.py's match threshold "
              f"({script.MATCH_THRESHOLD}) - this may not be a real match",
              file=sys.stderr)


if __name__ == "__main__":
    main()
