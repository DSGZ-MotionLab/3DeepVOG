"""
3DeepVOG CLI Tool - Eye Tracking Pipeline Configuration

This script is part of the 3DeepVOG project. It parses command-line arguments and sets up
configuration parameters for processing eye-tracking videos, including gaze estimation,
torsion tracking, and segmentation.

Features:
- CLI interface with argparse
- available device selection (MPS, CUDA, CPU)
- Video metadata extraction
- Configurable segmentation and torsion parameters

Author: Jingkang Zhao
Date: 2025-07-01
Version: 1.0.0

TODO:
- GUI interface for easier configuration (in clinical applications)
"""


from args_marker_v2 import make_args
from main import run_3deepvog, collect_results
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    args = make_args()
    args.update({
        # 'fit_vid': XXXX,   #comment out: use pred_vid for fitting eyeball model
        'pred_vid': r"D:\jzhao\DeepVOG-project\datasets\Patientrecording\sub3\trial.avi",
        # 'pred_vid': r"D:\jzhao\DeepVOG-project\video_test\test_cornea_el_fitting\ES_gaze_jzhao_v3.mp4",
        # 'is_parallel': True,   #True: multithread, False: single thread (sequential)
        # 'batch_size': 32,   #batch size for processing frames
        'device': 'cuda',   #comment out: use default device, e.g. 'cuda', 'mps', 'cpu'
        # # 'mode': 'fit',  # 'fit' or 'predict'
        # 'segmentation_model': xxx,   #e.g. 'SegResNet_3in3out'
        # 'segmentation_model_weights_path': xxx,  #TODO:  e.g. 'SegResNet_3in3out'
        # # 'max_frame': 1000,  # comment out: default is all frames
        # # 'eyeball_path': xxx,   #can define custom eyeball model path, comment out if not needed
        # 'extract_segment_map': False,  #"all", "sclera", "False"
        'do_gaze_tracking': True,
        # 'eyeball_model': 'simple', #simple or LeGrand or PL
        'do_torsion_tracking': True,
        # 'torsion_collecte_detail': False,
        # 'focal_length': 16.0,  # mm
        # 'sensor_size': (4.8, 3.6),  # mm
        # 'viz_segmentation': False,
        # 'viz_gaze': False,
        'viz_results': False,
    })

    threads, updated_params = run_3deepvog(args)
    df_ellipses, df_gaze_out, _, torsion_out, *_ = collect_results(threads, updated_params)
    # base_name = os.path.splitext(os.path.basename(args['pred_vid']))[0]
    # df_ellipses.to_csv(f"{base_name}_ellipses.csv", index=False)
    # df_gaze_out.to_csv(f"{base_name}_gaze.csv", index=False)