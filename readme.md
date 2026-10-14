# Sheep MOT Analysis

A pipeline for evaluating multiple object-tracking algorithms on sheep videos. It runs four trackers — SORT, BoTSORT, DeepSORT, and ByteTrack — and uses TrackEval to compare their tracking results against ground-truth annotations using MOT metrics.

## 1. Start Fresh (Optional)

If you want to run the project from scratch, delete the following generated files and folders first:

- `TrackEval/`
- `trackeval_outputs/`
- `sheep_results.xlsx`

**Warning:** These contain generated evaluation files and previous results. Back up anything you want to keep before deleting them.


## 2. Prepare Your Files

Before running the pipeline, make sure the following folders contain the required files:

- **Videos:** Place your input video in the `videos/` folder.
- **Ground truth:** Place your ground-truth ZIP file in the `groundtruth/` folder.
- **Models:** Pretrained YOLO models available for testing are stored in the `models/` folder.

## 3. Configure Your Settings

Open `settings.py` and configure the following parameters:

| Setting | Description |
|---|---|
| `video_name` | Name of the video you want to process. |
| `gt_zip` | Path to the ground-truth ZIP file corresponding to the selected video. |
| `seq_name` | Specify a sequence name, such as `"seq22"`. Set to `None` to automatically select the next available sequence. |
| `show_detections` | Set to `True` to view the trackers working on the video. |
| `run_eval` | Set to `True` to run all trackers and evaluate their outputs against the ground truth using TrackEval. Set to `False` to run tracking only. |
| `eval_only` | Set to `True` to evaluate previously generated tracker outputs without running the trackers again. |
| `detector_name` | Select a detector to test its performance. The project includes some locally available YOLO models in the `models/` folder. |

### Choosing a Detector

You can experiment with different YOLO models to compare their performance.

I have already used RF-DETR to generate the ground-truth annotations, so using RF-DETR again as the detector for this evaluation may not be useful for the intended comparison.

## 4. Generate Ground Truth

If you haven't generated ground-truth annotations for your video yet, use the linked Kaggle notebook:

**[kaggle_link](https://www.kaggle.com/code/neeharikajugran/sam2-yolo-cvat/edit)**

Make a copy of the notebook, upload your own video, and follow the instructions to generate ground-truth annotations.

The notebook uses Kaggle's GPU support, which can be helpful when processing videos.

Once the annotations are ready, place the resulting ground-truth ZIP file in the `groundtruth/` folder and update `gt_zip` in `settings.py`.

## 5. Run Your First Video

From the project root directory, execute:

```bash
python .\scripts\run_config.py
```

### First-time setup

When prompted:

**"Have you set up ground truth for this sequence?"**

Enter:

```text
n
```

This tells the script to set up the ground-truth data for the selected sequence.

During setup, the script prepares the required data inside the `TrackEval/` directory, including the ground-truth bounding-box files needed for evaluation.

You may be asked to provide the ground-truth ZIP path again. Enter the correct path and let the trackers run.

The pipeline will then run the configured trackers and, if `run_eval = True`, evaluate their outputs using TrackEval.

> **Note:** This setup step is required when running the project for the first time or when configuring a sequence whose ground truth has not yet been set up.

## 6. Fix NumPy or Data-Type Compatibility Errors

If TrackEval encounters errors such as `np.float` or other NumPy compatibility issues, run:

```bash
python fix_compatibility.py
```

Then update the settings in `settings.py`:

```python
eval_only = True
run_eval = False
```

Run the script again:

```bash
python .\scripts\run_config.py
```

This configuration is intended to evaluate existing tracker outputs without running the trackers again.

**Important:** Use evaluation-only mode only after the tracker output files have been generated successfully.

## 7. Export Evaluation Results to Excel

After TrackEval finishes, the evaluation logs are saved in the `trackeval_outputs/` folder.

To export the results to Excel, run:

```bash
python .\scripts\results_to_excel.py --folder "path_to_trackeval_output_folder"
```

Replace `path_to_trackeval_output_folder` with the actual path to your evaluation output folder.

For example, if the folder is in the project root:

```bash
python .\scripts\results_to_excel.py --folder ".\trackeval_outputs"
```

The script processes the evaluation result files and exports the results to an Excel workbook for easier comparison.

## 8. Typical Workflow

1. Place your video in `videos/`.
2. Generate ground-truth annotations and place the ZIP file in `groundtruth/`.
3. Configure `settings.py`.
4. Run `python .\scripts\run_config.py`.
5. Complete the ground-truth setup when prompted.
6. Let the trackers run and TrackEval calculate the metrics.
7. Export the evaluation results to Excel using `results_to_excel.py`.
8. Compare the performance of SORT, BoTSORT, DeepSORT, and ByteTrack.

## Notes

- Make sure the ground-truth annotations correspond to the video being evaluated.
- Use `eval_only = True` only when valid tracker outputs already exist.
- Keep previous evaluation logs and Excel workbooks if you need to preserve experiment history.
- When starting fresh, remove the generated folders and workbook listed in Section 1 only if you no longer need their contents.
