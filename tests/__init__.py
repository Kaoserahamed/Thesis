"""Test suite for the River Morphology Prediction thesis project.

Covers:
  * numpy metrics (IoU, Dice, area difference)
  * data-pipeline helpers (sequence generation, leakage-proof split)
  * data utilities (GeoTIFF I/O, normalisation, water masks)
  * visualisation helpers
  * model builders (TensorFlow-dependent — marked ``requires_tensorflow`` / ``slow``)
"""
