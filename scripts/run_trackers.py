import sys
import subprocess
from ultralytics import YOLO
import cv2
import numpy as np
import os

from sort_tracker import Sort
from fix_compatibiltiy import fix_trackeval_numpy
from deep_sort_realtime.deepsort_tracker import DeepSort

from pathlib import Path
import re
from settings import ( DETECTOR_NAME)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Move up one directory if needed
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

TRACKERS_DIR = Path(PROJECT_DIR) / "TrackEval" / "data" / "trackers" / "mot_challenge" / "Sheep-train"

TRACKER_NAMES = ["SORT", "BoTSORT", "DeepSORT", "ByteTrack"]

DETECTOR_PATH = os.path.join(PROJECT_DIR, "models", DETECTOR_NAME)

OUTPUTS_DIR = os.path.join(PROJECT_DIR, "trackeval_outputs")


def ensure_detector_weights():
    """Return the path to the detector weights, downloading them first if they are missing."""
    if os.path.isfile(DETECTOR_PATH):
        return DETECTOR_PATH

    name = os.path.basename(DETECTOR_PATH)
    print(f"Detector weights not found at {DETECTOR_PATH}. Downloading {name} ...")
    os.makedirs(os.path.dirname(DETECTOR_PATH), exist_ok=True)
    try:
        # Downloads official Ultralytics weights (yolov8m.pt, yolo11x.pt, ...) straight to DETECTOR_PATH
        from ultralytics.utils.downloads import attempt_download_asset
        path = attempt_download_asset(DETECTOR_PATH)
    except Exception as e:
        raise FileNotFoundError(
            f"Could not download {name}: {e}\n"
            f"Check your internet connection, or place the weights at {DETECTOR_PATH} manually."
        ) from e

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{name} is not an official Ultralytics weights file, so it can't be downloaded automatically.\n"
            f"Place it at {DETECTOR_PATH} manually."
        )
    print(f"Weights ready: {path}")
    return path


def load_detector():
    """Load a fresh YOLO model (fresh so tracker state never leaks between trackers)."""
    return YOLO(ensure_detector_weights())


def read_frames(video_path):
    """Yield (frame_number, frame), 1-based. Always releases the capture."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    try:
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            yield frame_count, frame
    finally:
        cap.release()
        cv2.destroyAllWindows()


def make_row(frame_count, x1, y1, x2, y2, track_id):
    return {
        'frame_num': frame_count,
        'x1': x1,
        'y1': y1,
        'w': x2 - x1,
        'h': y2 - y1,
        'track_id': int(track_id)
    }


def draw_box(frame, x1, y1, x2, y2, track_id):
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, f'Sheep id: {track_id}', (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def sort_tracker_det(video_path, show_detections=False):
    tracker_sort = Sort(max_age=90, min_hits=3, iou_threshold=0.3)
    yolo = load_detector()
    tracker_outputs = []

    for frame_count, frame in read_frames(video_path):
        yolo_results = yolo(frame, verbose=False)
        detections = np.empty((0, 5))
        for result in yolo_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                if yolo.names[int(box.cls[0])] == 'sheep':
                    detections = np.vstack((detections, [x1, y1, x2, y2, confidence]))

        tracker_result = tracker_sort.update(detections)
        for result in tracker_result:
            # works whether Sort returns [x1,y1,x2,y2,id] or [x1,y1,x2,y2,conf,id]
            x1, y1, x2, y2 = map(int, result[:4])
            track_id = int(result[-1])
            tracker_outputs.append(make_row(frame_count, x1, y1, x2, y2, track_id))
            if show_detections:
                draw_box(frame, x1, y1, x2, y2, track_id)

        if show_detections:
            cv2.imshow('SORT', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return tracker_outputs


def botsort_tracker_det(video_path, show_detections=False):
    yolo = load_detector()
    tracker_outputs = []

    for frame_count, frame in read_frames(video_path):
        yolo_results = yolo.track(frame, persist=True, tracker='botsort.yaml', verbose=False)

        for result in yolo_results:
            if result.boxes.id is None:
                continue

            for box, track_id, cls in zip(result.boxes.xyxy, result.boxes.id, result.boxes.cls):
                if yolo.names[int(cls)] != 'sheep':
                    continue

                x1, y1, x2, y2 = map(int, box.tolist())
                tracker_outputs.append(make_row(frame_count, x1, y1, x2, y2, track_id))
                if show_detections:
                    draw_box(frame, x1, y1, x2, y2, int(track_id))

        if show_detections:
            cv2.imshow('BoTSORT', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return tracker_outputs


def deepsort_tracker_det(video_path, show_detections=False):
    tracker_deepsort = DeepSort(max_age=90, n_init=3, max_iou_distance=0.7,
                                max_cosine_distance=0.3, embedder='mobilenet')
    yolo = load_detector()
    tracker_outputs = []

    for frame_count, frame in read_frames(video_path):
        yolo_results = yolo(frame, verbose=False)
        detections = []
        for result in yolo_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                confidence = float(box.conf[0])
                if yolo.names[int(box.cls[0])] == 'sheep':
                    # DeepSORT expects: ([left, top, width, height], confidence, class)
                    detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, 'sheep'))

        tracks = tracker_deepsort.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            tracker_outputs.append(make_row(frame_count, x1, y1, x2, y2, track.track_id))
            if show_detections:
                draw_box(frame, x1, y1, x2, y2, track.track_id)

        if show_detections:
            cv2.imshow('DeepSORT', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return tracker_outputs


def bytetrack_tracker_det(video_path, show_detections=False):
    yolo = load_detector()
    tracker_outputs = []

    for frame_count, frame in read_frames(video_path):
        yolo_results = yolo.track(frame, persist=True, tracker='bytetrack.yaml', verbose=False)

        for result in yolo_results:
            if result.boxes.id is None:
                continue

            for box, track_id, cls in zip(result.boxes.xyxy, result.boxes.id, result.boxes.cls):
                if yolo.names[int(cls)] != 'sheep':
                    continue

                x1, y1, x2, y2 = map(int, box.tolist())
                tracker_outputs.append(make_row(frame_count, x1, y1, x2, y2, track_id))
                if show_detections:
                    draw_box(frame, x1, y1, x2, y2, int(track_id))

        if show_detections:
            cv2.imshow('ByteTrack', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    return tracker_outputs


def write_mot_format(tracker_data, tracker_name, seq_name):
    output_dir = TRACKERS_DIR / tracker_name / "data"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{seq_name}.txt"

    if not tracker_data:
        print(f"WARNING: {tracker_name} produced no tracks for {seq_name}")

    lines = []
    for det in tracker_data:
        frame = int(det['frame_num'])
        track_id = int(det['track_id'])
        x, y, w, h = det['x1'], det['y1'], det['w'], det['h']
        lines.append(f"{frame},{track_id},{x},{y},{w},{h},1,1,1")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved: {output_path}")


def get_seq_nums(data_dir):
    """Return sequence numbers found in a tracker's data directory."""
    pattern = re.compile(r"seq(\d+)\.txt$", re.IGNORECASE)

    if not data_dir.exists():
        return set()

    seq_nums = set()

    for file_path in data_dir.glob("seq*.txt"):
        match = pattern.fullmatch(file_path.name)
        if match:
            seq_nums.add(int(match.group(1)))

    return seq_nums


def check_tracker_sequences(trackers_dir=TRACKERS_DIR):
    """Check sequence consistency across all four trackers. Returns next seq number or None."""
    trackers_dir = Path(trackers_dir)
    tracker_sequences = {}

    for tracker in TRACKER_NAMES:
        data_dir = trackers_dir / tracker / "data"
        tracker_sequences[tracker] = get_seq_nums(data_dir)

    all_sequences = set.union(*tracker_sequences.values())

    if not all_sequences:
        print("No sequence files found. Starting at seq01.")
        return 1

    # Check every sequence against every tracker
    mismatches = {}

    for seq_num in sorted(all_sequences):
        missing = [
            tracker
            for tracker, sequences in tracker_sequences.items()
            if seq_num not in sequences
        ]

        if missing:
            mismatches[seq_num] = missing

    if mismatches:
        print("WARNING: Sequence numbers do not match across all four trackers!")

        for seq_num, missing in mismatches.items():
            print(f"seq{seq_num:02d} is missing from: {', '.join(missing)}")

        print(
            "\nIncomplete sequences must be deleted from the relevant "
            "tracker folders before continuing."
        )
        return None

    latest_seq = max(all_sequences)
    next_seq = latest_seq + 1

    print("All four trackers have matching sequence numbers.")
    print(f"Latest sequence: seq{latest_seq:02d}")
    print(f"Next sequence:   seq{next_seq:02d}")

    return next_seq


def run_all_trackers(video_path, seq_name, show_detections=True):
    # Run every tracker first and write only if all succeed, so a failure
    # can't leave the tracker folders with mismatched sequences.
    tracker_outputs = {
        'SORT': sort_tracker_det(video_path, show_detections),
        'BoTSORT': botsort_tracker_det(video_path, show_detections),
        'DeepSORT': deepsort_tracker_det(video_path, show_detections),
        'ByteTrack': bytetrack_tracker_det(video_path, show_detections),
    }

    for name, data in tracker_outputs.items():
        write_mot_format(data, name, seq_name)


def run_trackeval(video_path, show_detections=True):
    if not os.path.isfile(video_path):
        print(f"Video not found: {video_path}")
        sys.exit(1)

    next_seq = check_tracker_sequences()
    if next_seq is None:
        print("Fix the sequence mismatch before running trackers.")
        sys.exit(1)

    seq_name = f"seq{next_seq:02d}"
    print("Ready to save results for:", seq_name)

    try:
        run_all_trackers(video_path, seq_name, show_detections=show_detections)
    except Exception as e:
        print(f"Tracking failed, no results were saved: {e}")
        sys.exit(1)

    evaluate_trackeval(video_path)


def evaluate_trackeval(video_path):
    """Run TrackEval on the tracker files already saved (no tracking). Saves the console log."""
    # TrackEval uses np.float / np.int, which newer NumPy removed (idempotent)
    try:
        fix_trackeval_numpy()
    except Exception as e:
        print(f"Could not apply TrackEval NumPy fixes: {e}")
        sys.exit(1)

    TRACK_EVAL_DIR = os.path.join(PROJECT_DIR, "TrackEval")

    cmd = [
        sys.executable,
        "scripts/run_mot_challenge.py",
        "--BENCHMARK", "Sheep",
        "--SPLIT_TO_EVAL", "train",
        "--TRACKERS_TO_EVAL", "SORT", "BoTSORT", "DeepSORT", "ByteTrack",
        "--METRICS", "HOTA", "CLEAR", "Identity",
        "--GT_FOLDER", "data/gt/mot_challenge/",
        "--TRACKERS_FOLDER", "data/trackers/mot_challenge/"
    ]

    video_name = Path(video_path).stem
    detector_name = Path(DETECTOR_PATH).stem
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    output_file = os.path.join(
        OUTPUTS_DIR, f"trackeval_output_for_{video_name}_{detector_name}.txt"
    )

    try:
        # Stream TrackEval's console output to the screen and to the file
        with open(output_file, "w", encoding="utf-8") as f, subprocess.Popen(
            cmd, cwd=TRACK_EVAL_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1
        ) as proc:
            for line in proc.stdout:
                print(line, end="")
                f.write(line)
    except FileNotFoundError:
        print(f"TrackEval not found at: {TRACK_EVAL_DIR}")
        sys.exit(1)
    except OSError as e:
        print(f"Could not save TrackEval output to {output_file}: {e}")
        sys.exit(1)

    print(f"TrackEval output saved to: {output_file}")

    if proc.returncode != 0:
        print(f"TrackEval failed with exit code {proc.returncode}")
        sys.exit(proc.returncode)


if __name__ == "__main__":
    video_path = sys.argv[1] if len(sys.argv) > 1 else "path/to/video.mp4"
    run_trackeval(video_path)