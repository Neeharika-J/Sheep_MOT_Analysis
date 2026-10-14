"""Master script. Edit the settings below, then run:  python run_config.py
Optionally override the video from the command line:  python run_config.py other

Steps: (0) auto first-time setup (clone TrackEval) -> (1) ground truth -> (2) trackers -> (3) optional TrackEval.
"""
import os
import re
import subprocess
import sys

from run_trackers import (PROJECT_DIR, check_tracker_sequences, run_all_trackers,
                          run_trackeval, evaluate_trackeval)
from setup import setup_trackeval_sequence, make_seqmap, TRACKEVAL_ROOT
from fix_compatibiltiy import fix_trackeval_numpy
from settings import (VIDEO_NAME, GT_ZIP, SEQ_NAME, SHOW_DETECTIONS, RUN_EVAL,
                      EVAL_ONLY)

VIDEOS_DIR = os.path.join(PROJECT_DIR, "videos")  # SHEEP_MOT_ANALYSIS/videos
GT_DIR = os.path.join(TRACKEVAL_ROOT, "data", "gt", "mot_challenge", "Sheep-train")
TRACKEVAL_REPO = "https://github.com/JonathonLuiten/TrackEval.git"


def ask_yes_no(question):
    while True:
        answer = input(f"{question} (y/n): ").strip().lower()
        if answer in ("y", "yes"):
            return True
        if answer in ("n", "no"):
            return False
        print("Please answer y or n.")


def clone_trackeval():
    """git clone TrackEval into PROJECT_DIR (skipped if it already exists)."""
    if os.path.isdir(TRACKEVAL_ROOT):
        print(f"TrackEval already exists at {TRACKEVAL_ROOT}, skipping clone.")
        return
    print(f"Cloning TrackEval into {TRACKEVAL_ROOT} ...")
    try:
        subprocess.run(["git", "clone", TRACKEVAL_REPO, TRACKEVAL_ROOT], check=True)
    except FileNotFoundError:
        print("git is not installed or not on PATH. Install git and try again.")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"git clone failed with exit code {e.returncode}")
        sys.exit(e.returncode)


def all_gt_sequences(extra=None):
    """Every seqNN folder already in the GT directory, plus `extra`."""
    seqs = set()
    if os.path.isdir(GT_DIR):
        seqs = {d for d in os.listdir(GT_DIR)
                if re.fullmatch(r"seq\d+", d) and os.path.isdir(os.path.join(GT_DIR, d))}
    if extra:
        seqs.add(extra)
    return sorted(seqs)


def run_ground_truth_setup(video_path, seq_name):
    """Run setup.py's functions for this sequence. Exits on failure."""
    zip_path = GT_ZIP or input("Path to the ground-truth ZIP: ").strip().strip('"').strip("'")
    try:
        setup_trackeval_sequence(video_path, zip_path, seq_name)
        make_seqmap(all_gt_sequences(extra=seq_name))
    except Exception as e:
        print(f"Ground-truth setup failed: {e}")
        sys.exit(1)


def ensure_ground_truth(video_path, seq_name):
    """Ask whether GT exists for this sequence; if not, set it up. Exits on failure."""
    gt_file = os.path.join(GT_DIR, seq_name, "gt", "gt.txt")

    if ask_yes_no(f"Have you already set up the ground truth for {seq_name}?"):
        if not os.path.isfile(gt_file):
            print(f"Ground truth not found at: {gt_file}")
            print(f"It must be set up under the name {seq_name} (the next free sequence).")
            sys.exit(1)
        return

    run_ground_truth_setup(video_path, seq_name)


def main():
    name = VIDEO_NAME
    if not name.lower().endswith(".mp4"):
        name += ".mp4"
    # bare/relative names are looked up in the videos folder; absolute paths are used as-is
    video_path = name if os.path.isabs(name) else os.path.join(VIDEOS_DIR, name)

    if not os.path.isfile(video_path):
        print(f"Video not found: {video_path}")
        sys.exit(1)

    # Re-run TrackEval on tracker files that are already saved (e.g. after a failed evaluation)
    if EVAL_ONLY:
        evaluate_trackeval(video_path)
        return

    # First-time setup is detected automatically: no TrackEval folder = not set up yet
    fresh_setup = not os.path.isdir(TRACKEVAL_ROOT)
    if fresh_setup:
        print("TrackEval not found, running first-time setup.")
        clone_trackeval()

    # TrackEval uses np.float / np.int, which newer NumPy removed (idempotent)
    try:
        fix_trackeval_numpy()
    except Exception as e:
        print(f"Could not apply TrackEval NumPy fixes: {e}")
        sys.exit(1)

    seq_name = SEQ_NAME
    if seq_name is None:
        next_seq = check_tracker_sequences()
        if next_seq is None:
            print("Fix the sequence mismatch before running trackers.")
            sys.exit(1)
        seq_name = f"seq{next_seq:02d}"

    if fresh_setup:
        # just cloned: go straight to running setup.py for the ground truth
        run_ground_truth_setup(video_path, seq_name)
    else:
        # already set up: just confirm the ground truth for this sequence
        ensure_ground_truth(video_path, seq_name)

    # run_trackeval re-picks the same sequence, runs all trackers, then TrackEval
    if RUN_EVAL and SEQ_NAME is None:
        run_trackeval(video_path)
        return

    print(f"Running all trackers on {video_path} -> {seq_name}")
    try:
        run_all_trackers(video_path, seq_name, show_detections=SHOW_DETECTIONS)
    except Exception as e:
        print(f"Tracking failed, no results were saved: {e}")
        sys.exit(1)

    if RUN_EVAL:
        print("RUN_EVAL with a fixed SEQ_NAME: run TrackEval manually or set SEQ_NAME = None.")


if __name__ == "__main__":
    main()