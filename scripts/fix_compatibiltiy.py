"""Patch TrackEval for newer NumPy (np.float / np.int / np.bool were removed).

Run on its own:  python fix_trackeval.py [path/to/TrackEval]
Or import:       from fix_trackeval import fix_trackeval_numpy
"""
import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
TRACKEVAL_ROOT = os.path.join(PROJECT_DIR, "TrackEval")


def fix_trackeval_numpy(trackeval_root=TRACKEVAL_ROOT):
    """Safe to run repeatedly: files are only rewritten when something changes.
    Word boundaries keep np.int32, np.float64, np.integer, etc. untouched.
    """
    root = os.path.join(trackeval_root, "trackeval")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"TrackEval source folder not found: {root}")

    patterns = [
        (re.compile(r"\bnp\.float\b"), "float"),
        (re.compile(r"\bnp\.int\b"), "int"),
        (re.compile(r"\bnp\.bool\b"), "bool"),
    ]

    fixed_files = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.endswith(".py"):
                continue
            path = os.path.join(dirpath, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    code = f.read()
                new_code = code
                for pattern, replacement in patterns:
                    new_code = pattern.sub(replacement, new_code)
                if new_code != code:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_code)
                    fixed_files.append(path)
            except OSError as e:
                print(f"WARNING: could not patch {path}: {e}")

    if fixed_files:
        print(f"TrackEval NumPy compatibility fixes applied to {len(fixed_files)} file(s).")
    else:
        print("TrackEval NumPy compatibility: nothing to fix.")
    return fixed_files


def main():
    # optional: python fix_trackeval.py path/to/TrackEval
    trackeval_root = sys.argv[1] if len(sys.argv) > 1 else TRACKEVAL_ROOT
    print(f"Patching TrackEval at: {trackeval_root}")
    try:
        fix_trackeval_numpy(trackeval_root)
    except FileNotFoundError as e:
        print(e)
        print("Clone TrackEval first (run run_config.py), then re-run this script.")
        sys.exit(1)
    except Exception as e:
        print(f"Failed to apply TrackEval NumPy fixes: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()