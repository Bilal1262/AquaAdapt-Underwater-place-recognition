# Limitations

- The final checkpoint is evaluated on one held-out trajectory, Fjord2. Because Fjord1 is
  included during training, this demonstrates held-out-trajectory robustness rather than
  completely unseen-environment generalization.
- Only 665 of 1,095 Fjord2 candidate queries have a valid geometric revisit after the
  10-second exclusion window. Retrieval metrics apply to those eligible queries, and
  coverage is reported separately.
- Clean Recall@1 improves by only 0.30 percentage points. The strongest result is the
  consistent advantage across all tested nonzero corruption severities.
- Calibration is inventoried, but the current raw-stream experiment does not rectify
  frames.
- Controlled low light, attenuation, haze, blur, and marine snow are reproducible
  robustness probes, not a complete physical underwater image-formation model.
- Gray-world/CLAHE/gamma enhancement is a simple comparison baseline, not a claim of
  state-of-the-art underwater enhancement.
- Official DINOv2 loading requires cached weights or initial network access.
- CPU execution is supported, but full bag extraction, descriptor encoding, and training
  are substantially faster with a CUDA-capable GPU.
