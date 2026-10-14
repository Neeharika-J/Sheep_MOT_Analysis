# SHEEP_MOT_ANALYSIS

Multi-object tracking of sheep in video. The project runs four trackers on the same video, scores them against hand-checked ground truth with [TrackEval](https://github.com/JonathonLuiten/TrackEval), and collects the scores in one Excel sheet so the trackers (and detectors) can be compared side by side.

## The idea behind this project...
While studying object detection i found most tutorials covered people, cars and other objects, but what if all objects were same, like boxes in a factory, or sheep in a farm. Then what if there are multiple similar-looking objects in a single frame. This motivated me to look for different types of trackers and detectors and understand how Multi-Object Tracking works.

## Why choose Sheep? 

Sheep are an interesting object detection experiment for me since all sheep look alike, they move in large numbers, and they don't move in constant speeds - some move faster than others, some stop midway and rest are slow. Their motion is hard to predict. Also creating groundtruth is not easy too. Looking for the right video for this project, I came across many videos where tracking sheeps manually was harder than I imagined.


!([Full Pipeline Demonstration](https://github.com/Neeharika-J/Sheep_MOT_Analysis/blob/main/images/Full_pipeline.png))



## SETUP:
There are 9 different detectors from the YOLO Family - Generations(YOLOV8, YOLOV10, YOLOV11), Sizes(n,l,x). There are 4 different trackers - SORT, BOTSORT, DEEPSORT, BYTETRACK - each having a unique architecture for detection, SORT being the simplest tracker. 

Every tracker was tested with the same set of detectors. This is important because some detectors may perform better than others. Since every tracker uses every detector, no tracker gets an unfair advantage from being tested only with a stronger detector.


## GROUNDTRUTH:
Instead of manual annotating the whole video, I've used a script that uses a combination of RF-DETR as detector and SAM2 as tracker.

### How does the notebook work?

1. Split the video into frames
The notebook breaks the video into individual images and makes them smaller so the computer can process them more easily.

2. Find the sheep
RF-DETR detects the sheep in the first image. If it finds three sheep, they are given IDs: Sheep 1, Sheep 2 and Sheep 3.

3. Follow the sheep
SAM2 tracks those sheep as they move through the video, keeping their IDs the same.

4. Check for new sheep every 30 frames
RF-DETR checks again. If it finds a sheep that wasn't being tracked before, the notebook gives it a new ID, such as Sheep 4.

5. Restart the tracker if a new sheep appears
SAM2 cannot simply add a new sheep to its existing tracking session. So the notebook restarts it, tells it about all the sheep found so far, and tracks them again up to the current frame.

6. Save the results
When the video is finished, the notebook restores the boxes to the original video size and saves them in a ZIP file that can be imported into CVAT for checking and correction.


### Why does it check every 30 frames?
Checking every frame with RF-DETR would take more time. Instead, SAM2 follows the sheep between checks, and RF-DETR looks for new sheep every 30 frames.
- Check less often: faster, but new sheep might be detected later.
- Check more often: slower, but new sheep can be detected sooner.


### What is the main problem to watch out for?
The computer can make mistakes. For example:
- It might miss a sheep.
- It might give a new sheep the wrong ID.
- It might accidentally swap the IDs of two sheep.
That's why reviewing the results in CVAT is necessary and correcting any mistakes before they are treated as accurate. 


## Trackers:

- SORT: A real-time tracking algorithm that uses Kalman filters to predict object positions and Hungarian algorithm-based matching to associate detections across frames, but is prone to ID switches during occlusion.

- BoT-SORT: An enhanced SORT-based tracker that combines Kalman filtering, improved data association, and optional appearance features to improve tracking accuracy, including in moving-camera scenarios.

- DeepSORT: Extends SORT by combining motion-based prediction with deep-learning-based appearance embeddings to maintain object identities through occlusions and overlapping trajectories.

- ByteTrack: Associates high-confidence detections first, then uses low-confidence detections to recover existing tracks, improving tracking continuity when objects are partially occluded or detected inconsistently.


## MOT-Metrics :
These are the mot metrics I used to analyse tracker performance:

#### 1. Detection Metrics

- DetA: Measures how accurately the tracker detects sheep, considering both missed sheep and incorrect detections. Higher is better.
- DetRe: Measures the proportion of real sheep that the tracker successfully detects. Higher is better.
- DetPr: Measures the proportion of predicted boxes that correctly contain sheep. Higher is better.

#### 2. Association Metrics

- AssA: Measures how accurately the tracker maintains the correct sheep identities across frames. Higher is better.
- AssRe: Measures how well the tracker maintains the identity of each real sheep throughout the video. Higher is better.
- AssPr: Measures how consistently each assigned ID belongs to the same sheep without switching to another. Higher is better.

#### 3. Overall Tracking and ID Metrics

- HOTA: Measures overall tracking performance by combining detection accuracy and identity association accuracy. Higher is better.
- IDF1: Measures how accurately the tracker identifies each sheep consistently throughout the video. Higher is better.
- IDSW: Counts how many times a sheep's assigned ID changes during tracking. Lower is better.
- GT_Det: Represents the total number of ground-truth sheep bounding boxes in the annotated video. It is a reference value, not a performance metric.
- |ID − GT_ID|: Measures the absolute difference between the number of predicted IDs and the actual number of sheep identities. Lower is better, with zero indicating equal counts.

## My General Findings:
### *Since its based on my limited research, these are not claims*

1. The sheep would move at different speeds causing other sheep to be partially or fully hidden. When that would happen, the detector bounding boxes would largely overlap and two or more sheep to be detected as one. This would lead to identity swtiches (IDSW metric).

2. The sheep look identical. So trackers like DeepSort which track based on appearance, create quite similar vector embeddings for each sheep, and is not able to tell them apart leading to lost tracks and identity switches (AssA and IDSW metrics).

3. A general COCO-trained detector like YOLO is not trained to specialize in sheep detection. The detector thus detect a rock as sheep or a far-away sheep as a rock in cases of slight ambiguity. RF-DETR is found to be more accurate than YOLO in this case and thus used as detector for groundtruth.

4. The video used in this experiments has camera movement, still ByteTrack is performing better than BotSort in many cases. BotSort is a slow, heavy tracker that uses mathematical computations to estimate camera motion. Bytetrack is lightweight and focuses on raw overlap (IoU) than camera movement. In a sheep herd, BotSort's Camera Compensation and Re-ID(Re Identification) overcomplicates its matching logic. ByteTrack is designed to recover obscured or blurred objects using low-confidence detections. Also the camera movements are not jittery allowing ByteTrack to simple do what it is designed to do.

5. Detection is an actual bottleneck in my case.


## Experiment Findings (Please refer notebooks\result_comparator.ipynb for this)

*All results come from one video. Finding mean of the trackers across all detectors*

### Tracker Comparison (averaged over all detectors)

| Tracker | HOTA mean (min to max) | IDF1 mean (min to max) | DetA | AssA | IDSW |
|---|---|---|---|---|---|
| **ByteTrack** | **57.7** (51.9 to 60.8) | **62.2** (54.4 to 66.0) | 47.9 | 69.6 | 2.7 |
| BoTSORT | 54.6 (48.8 to 57.4) | 57.4 (49.4 to 61.6) | 42.6 | 70.3 | 2.4 |
| SORT | 53.8 (47.2 to 57.8) | 60.9 (54.2 to 65.2) | 44.7 | 64.8 | 2.6 |
| DeepSORT | 47.5 (40.0 to 51.7) | 56.0 (43.1 to 63.0) | 43.8 | 51.7 | 15.0 |

*The min and max columns show the spread across detectors.*

ByteTrack has the highest mean HOTA (57.7) and the highest mean IDF1 (62.2). Its weakest result (HOTA 51.9) is still above DeepSORT's best (51.7), and its best (60.8) is the highest of any tracker.

IDF1 gives a slightly different picture for the middle of the ranking. SORT (60.9) beats BoTSORT (57.4) on IDF1 even though the two are nearly tied on HOTA (53.8 vs 54.6). DeepSORT is last on both metrics, but its IDF1 gap is smaller than its HOTA gap, and its IDF1 range is the widest (43.1 to 63.0), which means it depends heavily on which detector it is paired with.

#### Detection (DetA, DetRe, DetPr)

All trackers miss many sheep. They output about 880 to 1,184 boxes against 1,597 ground-truth boxes, which matches recall of only 47 to 51%.

- **BoTSORT** has the highest recall (51.4) and outputs the most boxes (1,184), but it has the lowest precision (69.4), so many of its extra boxes are false.
- **ByteTrack** has the highest precision (86.6) and the best DetA (47.9), so it makes the fewest false boxes while still finding a similar number of sheep.
- **SORT and DeepSORT** output fewer boxes. A likely reason is that they only confirm a track after several consecutive hits (`min_hits`) and discard weak detections, whereas ByteTrack and BoTSORT also use low-confidence detections. This is a hypothesis based on how the algorithms work, not something these results prove.

#### Association (AssA, IDSW)

- **BoTSORT (70.3) and ByteTrack (69.6)** are nearly tied, and a gap of 0.7 is too small to name a winner.
- **SORT (64.8)** is somewhat behind, with a similar number of ID switches to the top two.
- **DeepSORT (51.7)** struggles with identity. It averages 15 ID switches, compared with about 2.5 for the others, and has the lowest AssA.

 

- **ByteTrack:** best detection accuracy and best IDF1, with association comparable to BoTSORT. It is the strongest overall.
- **BoTSORT:** keeps identities well and finds the most sheep, but its false detections lower its overall score.
- **SORT:** reasonable identity keeping, with good IDF1, but behind the top two on association.
- **DeepSORT:** the weakest, with by far the most ID switches.

---
###  Detector Comparision

The table below shows which tracker performed best for each YOLO detector and how much it outperformed the second-best tracker in each case, based on HOTA.

| Detector | Best tracker | Margin over 2nd-best |
| -------- | ------------ | -------------------- |
| YOLO11l  | ByteTrack    | 3.9                  |
| YOLO11x  | ByteTrack    | 2.9                  |
| YOLO11n  | ByteTrack    | 3.1                  |
| YOLOv10l | ByteTrack    | 2.9                  |
| YOLOv10x | ByteTrack    | 4.1                  |
| YOLOv10n | ByteTrack    | 2.9                  |
| YOLOv8l  | ByteTrack    | 2.4                  |
| YOLOv8x  | ByteTrack    | 3.9                  |
| YOLOv8n  | ByteTrack    | 0.4                  |


- ByteTrack wins for all nine detectors, achieving the highest HOTA in every comparison.
- Its largest advantage is with YOLOv10x, where it scores 4.1 points higher than the second-best tracker.
- Its smallest advantage is with YOLOv8n, where the margin is only 0.4 points, making the result much closer.
- The results suggest ByteTrack is the strongest overall tracking choice for this experiment, although the small margin for YOLOv8n means its advantage there is less pronounced.

| Version | n     | l     | x     |
| ------- | ----- | ----- | ----- |
| YOLO11  | 51.90 | 60.83 | 60.70 |
| YOLOv10 | 53.02 | 60.31 | 60.58 |
| YOLOv8  | 52.26 | 59.33 | 60.45 |

The results show that model size has a greater impact on HOTA than YOLO version. Large and extra-large models consistently outperform nano models, with HOTA scores around 59–61 compared to 52–53 for nano models. YOLO11l achieves the highest HOTA (60.83), while performance differences between YOLO versions are relatively small. Overall, large models offer a good balance of tracking accuracy and model size.

These findings are supported by the graphs,

![](SHEEP_MOT_ANALYSIS\images\trackers.png)

![](SHEEP_MOT_ANALYSIS\images\detectors.png)


Finally, all trackers output fewer boxes than the 1,597 ground-truth boxes for two reasons:
1. **Detectors miss sheep.** The ground truth labels every partly visible sheep (built with RF-DETR + SAM2), while pretrained YOLO models skip small, occluded, or cut-off ones. Recall is only about 47 to 51% for this reason.
2. **Trackers filter further.** SORT and DeepSORT only report a track after several consecutive hits (`min_hits`) and drop weak detections. ByteTrack and BoTSORT also use low-confidence detections, which is why they output more boxes (BoTSORT 1,184 vs SORT 881).


![](SHEEP_MOT_ANALYSIS\images\detections.png)


## Limitations and Future Work

- Only one video was tested. The results may differ with changes in lighting, camera angle, or the number of sheep. More videos are needed to confirm the findings.

- Ground truth may contain bias. The reference annotations were created using RF-DETR and SAM2, then corrected manually in CVAT. This may favor similar detection methods. Since there are only 1,597 annotated boxes, small differences of around 1 HOTA point may not be significant.

- Default settings were used. The detectors and trackers were not fine-tuned. Recall was around 50%, meaning many sheep were missed. Tuning the settings could improve performance, particularly for DeepSORT and SORT.

- Future work. Fine-tune the detector using annotated sheep images, optimize tracker settings, compare processing speed (FPS), and test tracking performance when sheep overlap or gather in crowded groups.
