# Limitations

- One `mclab_1` trajectory supports development, not cross-environment generalization.
- Place recognition needs revisits after temporal exclusion; low coverage is possible and
  should motivate adding `mclab_2`, not relaxing ground truth after observing results.
- Calibration is inventoried, but initial raw-stream experiments do not rectify frames.
- Underwater corruptions are lightweight controlled probes, not exact physical simulation.
- Gray-world/CLAHE/gamma enhancement is a simple reproducible comparison baseline.
- Official DINOv2 loading requires cached weights or initial network access.
- CPU execution is supported but full descriptor extraction and training can be slow.

