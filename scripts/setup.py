# git clone https://github.com/JonathonLuiten/TrackEval.git

import os
import cv2
import zipfile
import shutil
import tempfile

# Same convention as run_trackers.py so both scripts point at the same TrackEval folder
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TRACKEVAL_ROOT = os.path.join(PROJECT_DIR, "TrackEval")


def setup_trackeval_sequence(video_path, groundtruth_zip, seq_name,
                             trackeval_root=TRACKEVAL_ROOT):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(groundtruth_zip):
        raise FileNotFoundError(f"Ground-truth ZIP not found: {groundtruth_zip}")
    if not os.path.isdir(trackeval_root):
        raise FileNotFoundError(
            f"TrackEval folder not found: {trackeval_root} "
            "(clone it with: git clone https://github.com/JonathonLuiten/TrackEval.git)"
        )

    base_gt = os.path.join(trackeval_root, "data", "gt", "mot_challenge", "Sheep-train", seq_name)
    gt_folder = os.path.join(base_gt, "gt")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()

    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"Invalid video metadata (fps={fps}, frames={frame_count}, "
            f"size={width}x{height}) for: {video_path}"
        )

    # Extract into a temp dir that is always cleaned up, and validate
    # everything before creating any output folders.
    with tempfile.TemporaryDirectory(prefix=f"temp_{seq_name}_") as temp_folder:
        try:
            with zipfile.ZipFile(groundtruth_zip, "r") as zip_ref:
                zip_ref.extractall(temp_folder)
        except zipfile.BadZipFile:
            raise ValueError(f"Not a valid ZIP file: {groundtruth_zip}")

        gt_source = os.path.join(temp_folder, "gt", "gt.txt")

        if not os.path.exists(gt_source):
            raise FileNotFoundError("gt/gt.txt was not found inside the ground-truth ZIP.")

        if os.path.exists(base_gt):
            print(f"WARNING: {base_gt} already exists and will be overwritten.")

        os.makedirs(gt_folder, exist_ok=True)
        shutil.copy(gt_source, os.path.join(gt_folder, "gt.txt"))

    # No indentation: configparser treats indented lines as continuations
    seqinfo = (
        "[Sequence]\n"
        f"name={seq_name}\n"
        "imDir=img1\n"
        f"frameRate={fps:g}\n"
        f"seqLength={frame_count}\n"
        f"imWidth={width}\n"
        f"imHeight={height}\n"
        "imExt=.jpg\n"
    )

    with open(os.path.join(base_gt, "seqinfo.ini"), "w") as f:
        f.write(seqinfo)

    print(f"{seq_name} setup complete!")
    print(f"FPS: {fps:g}")
    print(f"Frames: {frame_count}")
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"Ground truth: {os.path.join(gt_folder, 'gt.txt')}")
    print(f"Sequence info: {os.path.join(base_gt, 'seqinfo.ini')}")


# setup_trackeval_sequence(normal, groundtruth_normal, "seq02") # change sequence here each groundtruth gets its own seq num


def make_seqmap(sequences, benchmark="Sheep", split="train",
                trackeval_root=TRACKEVAL_ROOT):
    """Create the seqmap file TrackEval needs.

    sequences: list of sequence names, e.g. ["seq01"]
    Writes: <root>/data/gt/mot_challenge/seqmaps/<benchmark>-<split>.txt
    """
    if not sequences:
        raise ValueError("sequences must be a non-empty list of sequence names")

    seqmap_dir = os.path.join(trackeval_root, "data", "gt", "mot_challenge", "seqmaps")
    os.makedirs(seqmap_dir, exist_ok=True)

    path = os.path.join(seqmap_dir, f"{benchmark}-{split}.txt")
    with open(path, "w") as f:
        f.write("name\n")                       # required header line
        f.write("\n".join(sequences) + "\n")    # one sequence per line

    print(f"{os.path.basename(path)} created at {path}")
    return path


# make_seqmap(["seq01", "seq02", "seq03"])

# make_seqmap(["seq01"])
