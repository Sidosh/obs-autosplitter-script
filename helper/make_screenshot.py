"""Standalone helper: screenshot a selectable area of the OBS Virtual Camera.

Start the virtual camera in OBS first (or load script.py, which starts it),
then run this with the same Python OBS uses so the packages installed by
script.py are available:
    C:\\Python311\\python.exe helper\\make_screenshot.py

Keys in the preview window:
    SPACE / ENTER  freeze the current frame and drag a rectangle over the
                   area to capture (ENTER confirms, c cancels)
    ESC / q        quit

Each confirmed selection is saved as a PNG (default: a screenshots folder
next to this script, override with --out-dir) and its coordinates are
printed as {"x": ..., "y": ..., "width": ..., "height": ...} for use in a
games/<game>/split.json entry.

Passing an area directly captures it and exits without opening any windows:
    C:\\Python311\\python.exe helper\\make_screenshot.py <x> <y> <width> <height>
"""

import argparse
import datetime
import os
import sys

try:
    import cv2
except ImportError:
    sys.exit("opencv-python is not installed for this interpreter. "
             "Install it with:\n"
             f"    {sys.executable} -m pip install --user opencv-python")

PROBE_MAX_INDEX = 10
PREVIEW_WINDOW = "OBS Virtual Camera  |  SPACE: select area   Q/ESC: quit"
SELECT_WINDOW = "Drag the area, then ENTER to save (c cancels)"
WINDOW_SIZE = (1280, 720)  # 16:9; frames are letterboxed if they differ
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


def save_crop(frame, x, y, w, h, out_dir):
    """Save the area of the frame as a timestamped PNG. Returns success."""
    frame_h, frame_w = frame.shape[:2]
    if x < 0 or y < 0 or w <= 0 or h <= 0 \
            or x + w > frame_w or y + h > frame_h:
        print(f"Area x={x} y={y} {w}x{h} is outside the "
              f"{frame_w}x{frame_h} frame", file=sys.stderr)
        return False

    os.makedirs(out_dir, exist_ok=True)
    name = datetime.datetime.now().strftime("screenshot_%Y%m%d_%H%M%S.png")
    path = os.path.join(out_dir, name)
    if not cv2.imwrite(path, frame[y:y + h, x:x + w]):
        print(f"Could not write {path}", file=sys.stderr)
        return False
    print(f"Saved {path}")
    print(f'  area: {{"x": {x}, "y": {y}, "width": {w}, "height": {h}}}')
    return True


def grab_frame(cap):
    """Read one frame after a few warm-up reads, or None on failure."""
    frame = None
    for _ in range(WARMUP_FRAMES):
        ok, frame = cap.read()
        if not ok:
            return None
    return frame


def select_and_save(frame, out_dir):
    """Let the user drag a rectangle over the frozen frame and save the crop.

    The window is scaled down to WINDOW_SIZE, but HighGUI maps mouse
    positions back to image coordinates, so the selection is still in
    full frame resolution.
    """
    cv2.namedWindow(SELECT_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(SELECT_WINDOW, *WINDOW_SIZE)
    x, y, w, h = cv2.selectROI(SELECT_WINDOW, frame,
                               showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(SELECT_WINDOW)
    if w == 0 or h == 0:
        print("Selection cancelled")
        return
    save_crop(frame, x, y, w, h, out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Save screenshots of a selectable area of the "
                    "OBS Virtual Camera.")
    parser.add_argument(
        "--index", type=int, default=None,
        help="cv2 camera index (default: find the OBS Virtual Camera "
             "by device name)")
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "screenshots"),
        help="directory for saved screenshots (default: %(default)s)")
    parser.add_argument(
        "area", nargs="*", type=int, metavar="N",
        help="x y width height: immediately screenshot that area and "
             "exit, without opening any windows")
    args = parser.parse_args()
    if args.area and len(args.area) != 4:
        parser.error("the area needs exactly 4 numbers: x y width height")

    index = args.index if args.index is not None else find_camera_index()
    if index < 0:
        sys.exit(1)

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(f"Could not open camera index {index}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if args.area:
        frame = grab_frame(cap)
        cap.release()
        if frame is None:
            sys.exit("Could not read a frame (virtual camera stopped?)")
        x, y, w, h = args.area
        if not save_crop(frame, x, y, w, h, args.out_dir):
            sys.exit(1)
        return

    print(f"Camera {index} open at {width}x{height}. "
          "SPACE/ENTER selects an area, Q/ESC quits.")

    # WINDOW_NORMAL makes the window resizable (16:9 default size) instead
    # of auto-sizing to the raw frame; the image keeps its aspect ratio.
    cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(PREVIEW_WINDOW, *WINDOW_SIZE)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost the camera feed (virtual camera stopped?)",
                      file=sys.stderr)
                break
            cv2.imshow(PREVIEW_WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC
                break
            if key in (32, 13):  # SPACE, ENTER
                select_and_save(frame, args.out_dir)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
