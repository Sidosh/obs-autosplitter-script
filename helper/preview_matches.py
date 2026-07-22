"""Standalone helper: live-preview matchTemplate scores for a category.

Start the virtual camera in OBS first (or load script.py, which starts
it), then run this with the same Python OBS uses so the packages
installed by script.py are available:
    C:\\Python311\\python.exe helper\\preview_matches.py "<Game>" "<Category>"

Opens a window over the camera feed listing every image referenced by
that category's splits.json, grouped by split key in the order script.py
processes them ('start', numeric keys ascending, 'stop'), each with its
live matchTemplate score. Green means the image currently scores at or
above the match threshold (i.e. script.py would count it as detected
right now); red means it doesn't. Use --threshold to try a different
cutoff than script.py's default without editing any code.

Keys in the preview window:
    ESC / q   quit
"""

import argparse
import json
import os
import sys

try:
    import cv2
    import numpy as np
except ImportError:
    sys.exit("opencv-python is not installed for this interpreter. "
             "Install it with:\n"
             f"    {sys.executable} -m pip install --user opencv-python")

GAMES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "games")
SPLITS_FILENAME = "splits.json"
DEFAULT_THRESHOLD = 0.9  # keep in sync with script.py's MATCH_THRESHOLD
WINDOW = "matchTemplate preview  |  Q/ESC: quit"
WINDOW_SIZE = (1280, 720)
LINE_HEIGHT = 22
MATCH_COLOR = (0, 220, 0)      # BGR
NO_MATCH_COLOR = (0, 0, 220)   # BGR


def find_camera_index():
    """cv2 device index of the OBS Virtual Camera, or -1 if not found."""
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


def load_template(path):
    """Reads an image file as a cv2 (BGR) array.

    cv2.imread() silently fails on Windows for paths containing
    non-ASCII characters (e.g. accented game names), so the file is
    read as bytes and decoded instead.
    """
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def ordered_split_keys(splits):
    """['start', then numeric keys ascending, then 'stop']; same order
    script.py's autosplit_worker processes them in."""
    def sort_key(key):
        if key == "start":
            return (0, 0)
        if key == "stop":
            return (2, 0)
        return (1, int(key))
    return sorted(splits.keys(), key=sort_key)


def validate_roi(label, roi):
    if (not isinstance(roi, (list, tuple)) or len(roi) != 4
            or not all(isinstance(n, int) for n in roi)):
        raise ValueError(
            f"'{label}' has an invalid roi (expected [x, y, w, h] ints): {roi!r}")


def parse_image_entry(key, entry):
    """One "images" list entry for split `key`: either a filename (matched
    against the full frame) or {filename: [x, y, w, h]} restricting
    matchTemplate to that rect just for this image; same format script.py
    reads. Returns (filename, roi_or_None)."""
    if isinstance(entry, str):
        return entry, None
    if isinstance(entry, dict) and len(entry) == 1:
        (name, roi), = entry.items()
        validate_roi(f"{key}: {name}", roi)
        return name, roi
    raise ValueError(f"'{key}' has an invalid images entry: {entry!r}")


def load_entries(category_dir):
    """[(label, template, roi), ...] for every image in splits.json, in
    split order. `roi` is None for images matched against the full frame."""
    path = os.path.join(category_dir, SPLITS_FILENAME)
    with open(path, "r", encoding="utf-8") as f:
        splits = json.load(f)

    entries = []
    for key in ordered_split_keys(splits):
        for entry in splits[key]["images"]:
            name, roi = parse_image_entry(key, entry)
            template = load_template(os.path.join(category_dir, "images", name))
            entries.append((f"[{key}] {name}", template, roi))
    return entries


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


def score(frame, template, roi=None):
    """Best matchTemplate score of `template` in `frame` (or within `roi`
    if given), or None if the search area is too small for the template."""
    if roi is not None:
        frame = crop_roi(frame, roi)
        if frame is None:
            return None
    th, tw = template.shape[:2]
    fh, fw = frame.shape[:2]
    if th > fh or tw > fw:
        return None
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    return float(result.max())


def draw_overlay(frame, entries, threshold):
    # Scored against the clean frame, before the text list darkens the
    # top-left corner below, since some ROIs sit in that same corner.
    scores = [score(frame, template, roi) for _, template, roi in entries]

    overlay = frame.copy()
    height = LINE_HEIGHT * len(entries) + 10
    width = max((cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                 for label, _, _ in entries), default=200) + 140
    cv2.rectangle(overlay, (0, 0), (min(width, frame.shape[1]), height),
                 (0, 0, 0), cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, dst=frame)

    for i, ((label, _, roi), s) in enumerate(zip(entries, scores)):
        y = LINE_HEIGHT * i + 18
        if s is None:
            text = f"{label}: template/roi larger than frame"
            color = NO_MATCH_COLOR
        else:
            text = f"{label}: {s:.3f}"
            color = MATCH_COLOR if s >= threshold else NO_MATCH_COLOR
        cv2.putText(frame, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   color, 1, cv2.LINE_AA)
        if roi is not None:
            rx, ry, rw, rh = roi
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), color, 1)


def list_games():
    if not os.path.isdir(GAMES_DIR):
        return []
    return sorted(n for n in os.listdir(GAMES_DIR)
                  if os.path.isdir(os.path.join(GAMES_DIR, n)))


def list_categories(game):
    game_dir = os.path.join(GAMES_DIR, game)
    if not os.path.isdir(game_dir):
        return []
    return sorted(n for n in os.listdir(game_dir)
                  if os.path.isdir(os.path.join(game_dir, n)))


def main():
    parser = argparse.ArgumentParser(
        description="Live-preview matchTemplate scores for a game's category.")
    parser.add_argument("game", help="folder name under games/, e.g. 'Pokémon Red-Blue'")
    parser.add_argument("category", help="folder name under games/<game>/, e.g. 'Catch Em All'")
    parser.add_argument("--index", type=int, default=None,
                       help="cv2 camera index (default: find the OBS Virtual "
                            "Camera by device name)")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                       help="match score to color green (default: %(default)s)")
    args = parser.parse_args()

    category_dir = os.path.join(GAMES_DIR, args.game, args.category)
    if not os.path.isdir(category_dir):
        games = list_games()
        categories = list_categories(args.game)
        parser.error(
            f"no such category '{args.game}/{args.category}'.\n"
            f"  games available: {games}\n"
            f"  categories available for '{args.game}': {categories}")

    try:
        entries = load_entries(category_dir)
    except (OSError, ValueError, KeyError) as e:
        sys.exit(f"Could not load splits for '{args.game}/{args.category}': {e}")
    if not entries:
        sys.exit(f"'{args.game}/{args.category}' has no images to preview")

    index = args.index if args.index is not None else find_camera_index()
    if index < 0:
        sys.exit(1)

    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        sys.exit(f"Could not open camera index {index}")

    print(f"Previewing {len(entries)} image(s) for "
          f"'{args.game}/{args.category}' at threshold {args.threshold}. "
          "Q/ESC quits.")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, *WINDOW_SIZE)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Lost the camera feed (virtual camera stopped?)",
                      file=sys.stderr)
                break
            draw_overlay(frame, entries, args.threshold)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # ESC
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
