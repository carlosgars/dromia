# Paper implementation

DromIA implements the paper's selected F11 + R1 system: independent-frame SAM 3.1
segmentation, PMPose-B, temporal bilateral identity correction, shoe assignment and ankle
refinement, bounded CoTracker/interpolation repair, and gait analysis from the active pose
revision and cached shoe evidence.

The expert loop creates immutable pose revisions. Metrics reference exactly one pose revision,
timebase, ground calibration, and configuration fingerprint; changing any input marks them
stale until **Generate Metrics** is selected again.

Experiments, ablations, alternative models, metric editing, full-sequence smoothing, and
legacy AIRUN compatibility are intentionally outside this repository.
