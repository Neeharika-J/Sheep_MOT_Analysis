# ---------------- SETTINGS ----------------
VIDEO_NAME = "sheep.mp4"               # file name inside SHEEP_MOT_ANALYSIS/videos (.mp4 added if missing)
GT_ZIP = 'groundtruth/sheep.zip'                      # ground-truth ZIP path; None = ask when needed
SEQ_NAME = None                    # e.g. "seq02"; None = auto-pick the next free sequence
SHOW_DETECTIONS = True             # show the live preview window (press q to skip a tracker)
RUN_EVAL = True                   # True = also run TrackEval afterwards
EVAL_ONLY = False
DETECTOR_NAME = "yolov8n.pt"       # name of the detector to use (include extentsion, e.g. .pt or .p)
# True = skip tracking, only (re)run TrackEval on saved tracker files
# ------------------------------------------