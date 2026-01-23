"""
3DeepVOG CLI Tool - Eye Tracking Pipeline Configuration

This script is part of the 3DeepVOG project. It parses command-line arguments and sets up
configuration parameters for processing eye-tracking videos, including gaze estimation,
torsion tracking, and segmentation.

Features:
- available device selection (CUDA -> works, CPU -> works, MPS-> not fully tested)
- Video metadata extraction
- Configurable segmentation and torsion parameters

Author: Jingkang Zhao
Date: 23/Jan/2026
Version: 2.0.0

TODO:
- CLI interface with argparse (in progress)
- GUI interface for easier configuration (in clinical applications)
"""

from threedeepvog.main import run_3deepvog

# import debugpy
# debugpy.listen(5678)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# print("Debugger attached - starting program")
if __name__ == '__main__':
    args = {
        # 'fit_vid': XXXX,   #comment out: if use pred_vid for fitting eyeball model
        'pred_vid': r"D:\jzhao\DeepVOG-project\3DeepVOG\sample_video\ES_gaze_320x240_pad.mp4",
        # 'pred_vid': r"D:\jzhao\DeepVOG-project\datasets\Gaze_EyeSeeCam\dataset2024\ekierig\trial.mp4",
        # 'is_parallel': True,   #True: multithread, False: single thread (sequential)
        # 'batch_size': 32,   #batch size for processing frames
        'device': 'cuda',   #comment out: use default device, e.g. 'cuda', 'mps', 'cpu'
        'model_frozen': True,  
        'segmentation_model': 'SegResNet_3in3out',   #e.g. 'SegResNet_3in3out'-> faster / 'SegFormerB0_3in3out' -> better accuracy
        # # 'max_frame': 1000,  # comment out: default is all frames
        'eyeball_model': 'PL', #simple or LeGrand or PL
        'connected_components': 'morph',   #e.g. False / 'morph'-> faster / 'largest'-> better accuracy
        'write_seg_video_type': 'processed',   # e.g. 'processed', 'raw'
        'focal_length': 16.0,  # mm scale: if not provided, will use default value, which may not be accurate
        'sensor_size': (4.8, 3.6),  # mm scale: if not provided, will use default value, which may not be accurate
        'gaze_tracking_flag': True,
        'torsion_tracking_flag': True,
        'seg_video_flag': True,   #record segmentation video
        'fit_video_flag': True,   #record fitting video -> can slow down processing
        # 'log_dir':  r"D:\jzhao\DeepVOG-project\video_test\segmentation_fMRI_VOG\log",  # set designated log dir
        'eyeball_path': None,   #can define custom eyeball model path, comment out if not needed
        'mode': 'auto'   #fit / predict / auto / all
    }

    run_3deepvog(active_args = args)
 