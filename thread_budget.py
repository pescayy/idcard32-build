# -*- coding: utf-8 -*-
"""One global thread budget for all model inference (batch 1118).

Before this module the budgets were set per library and multiplied:
the SCRFD detection session took cpu-1 threads, each of RapidOCR's
three sessions defaulted to all cores, and OpenCV ran its own
all-cores pool -- on Qing's 4-core Win7 machine (and in batch runs)
the pools fought each other instead of doing work. Now:

  BUDGET          -- every inference session gets this many intra-op
                     threads: all cores but one (the reserved core
                     keeps the UI responsive, the 2026-11-12 rule).
  INFERENCE_LOCK  -- one process-wide lock serialising every model
                     run (card detection, licence detection, OCR), so
                     two sessions can never be resident at once and
                     the budget is never multiplied by concurrency.
                     Callers queue strictly; nothing is dropped.
"""
import os
import threading

BUDGET = max(1, (os.cpu_count() or 2) - 1)

INFERENCE_LOCK = threading.Lock()


def apply_cv2_budget():
    """Cap OpenCV's own thread pool at the same budget. Idempotent;
    called from the app entry and the OCR/detection constructors'
    modules so headless pipeline use is covered too."""
    try:
        import cv2
        cv2.setNumThreads(BUDGET)
    except Exception:
        pass
