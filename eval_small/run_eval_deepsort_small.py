import runpy
import sys

import numpy as np

if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)

sys.path.insert(0, "eval_small")
sys.argv = [
    "eval.py",
    "eval_small/gt_train_c001_c002_f001_120.txt",
    "eval_small/pred_deepsort_mask_rcnn_c001_c002_f001_120.txt",
    "--dstype",
    "train",
    "--roidir",
    "eval_small/ROIs",
]

runpy.run_path("AICity22_Track1_MTMC_Tracking/eval/eval.py", run_name="__main__")
