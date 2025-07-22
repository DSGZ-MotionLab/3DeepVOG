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

NOTE:
another good CLI tool: python-fire (https://github.com/google/python-fire)

TODO:
- GUI interface for easier configuration (in clinical applications)
"""

from pprint import pprint
import argparse
import os, platform, torch, sys
import numpy as np
from utils.read_and_save import get_video_info_torch


def get_conf_args(active_args):
    """
    Given user arguments, extract video metadata and construct a dictionary of config parameters.

    Returns:
        dict: Configuration with video info, derived paths, and default values.
    """
    # --- Setup ---
    device = torch.device("mps") if torch.backends.mps.is_available() \
        else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    
    if active_args['mode'] == 'predict':
        ff_input_vid = active_args['pred_vid']
    else:
        ff_input_vid = active_args['fit_vid']

    OS_name = 'macOS' if platform.system() == 'Darwin' else platform.system()
    # --- Extract video info ---
    print('\n')
    print(f"**********  START PROCESSING ({active_args['mode']} mode) **********")
    # print(f"{active_args['mode']} mode...")
    print(f"Video source: {ff_input_vid}")
    print(f"OS: {OS_name}")
    print(f"Device: {active_args.get('device', device)}")
    print(f"Running in {'parallel (multi-thread)' if active_args['is_parallel'] else 'sequential (single-thread)'} mode")

    # --- Video Input ---
    vid_save_path = ff_input_vid.replace('.'+ff_input_vid.split('.')[-1],'_dv3dviz.mp4')
    if not ff_input_vid or not os.path.exists(ff_input_vid):
        print(f"Video file does not exist 🥺")
        sys.exit(1)  # Exit the script with error code 1

    vid_root, vid_name = os.path.split(ff_input_vid)
    vid_base_name, vid_ext = os.path.splitext(vid_name)

    vid_info = get_video_info_torch(ff_input_vid)
    vid_name_root, ext, vid_reader, \
    (vid_nr_frames, vid_h, vid_w, vid_channels), \
        vid_shape_src, vid_img_scaling_factor, vid_fps = vid_info
    print(f"Video info: {vid_h}x{vid_w}, {vid_fps} fps, {vid_nr_frames} frames.")
    print('***************************************************\n')

    # --- Param Defaults ---
    conf_args = {
        'OS': OS_name,
        'vid_h': vid_h,
        'vid_w': vid_w,
        'vid_name_root': vid_name_root,
        'vid_reader': vid_reader,
        'resolution': (vid_w, vid_h),
        'vid_channels': vid_channels,
        'vid_fps': vid_fps,
        'vid_timestep': 1 / vid_fps,
        'vid_nr_frames': vid_nr_frames,
        'vid_img_scaling_factor': vid_img_scaling_factor,
        'ff_input_vid': ff_input_vid,
        'vid_root': vid_root,
        'vid_name': vid_name,
        'vid_base_name': vid_base_name,
        'vid_ext': vid_ext,
        'early_stop': False,
        'max_frame': active_args.get('max_frame', vid_nr_frames),  # None means all frames by default
        'viz_filename_mp4': vid_save_path,
        'viz_filename_mp4_ellipses': vid_save_path.replace('.mp4', '_visualization.mp4'),
    }
    default_eyeball_path = os.path.join(conf_args['vid_root'], 
                            f"{conf_args['vid_base_name']}_{active_args['eyeball_model']}_eyeball_model.json")
    conf_args['eyeball_path'] = active_args.get('eyeball_path', default_eyeball_path)
    return conf_args


def make_args():
    parser = argparse.ArgumentParser(description="Eye Tracking Pipeline Configuration")
    parser.add_argument('--version', action='version', version='DeepVOG CLI 1.0.0')

    # --- Required / Core Inputs ---
    parser.add_argument('--pred_vid', type=str, default="xxx",
                        help="Path to the video used for prediction")

    parser.add_argument('--fit_vid', type=str, default=argparse.SUPPRESS,
                        help="Path to the video used for eyeball fitting; defaults to pred_vid if not provided")

    parser.add_argument('--mode', type=str, choices=["fit", "predict", "auto"], default= "auto",
                        help="'fit', 'predict', or 'auto' to choose processing mode")

    # --- Runtime / Device ---
    parser.add_argument('--is_parallel', type=str, default="False",
                        help="Enable parallel (multi-threaded) processing")
    parser.add_argument('--device', type=str, choices=["cpu", "cuda", "mps"], default="cpu",
                        help="Compute device to use: 'cpu', 'cuda', or 'mps'")

    # --- Segmentation ---
    parser.add_argument('--segmentation_model', type=str,
                        help="Name of the segmentation model to use (e.g., 'SegResNet_3in3out')")

    parser.add_argument('--segmentation_model_weights_path', type=str,
                        help="Path to pre-trained segmentation model weights")

    # --- Gaze & Eyeball ---
    parser.add_argument('--extract_segment_map', type=str, default=argparse.SUPPRESS,
                        help="Which segmentation maps to extract: 'all', 'sclera', or 'False'")

    parser.add_argument('--do_gaze_tracking', action='store_true',
                        help="Enable gaze tracking")

    parser.add_argument('--eyeball_model', type=str, default="PL", choices=["simple", "LeGrand", "PL"],
                        help="Eyeball model type: 'simple', 'LeGrand', or 'PL'")

    parser.add_argument('--eyeball_path', type=str, default=argparse.SUPPRESS,
                        help="Path to custom eyeball model JSON file")

    # --- Torsion ---
    parser.add_argument('--do_torsion_tracking', action='store_true',
                        help="Enable torsion tracking")

    parser.add_argument('--torsion_collecte_detail', action='store_true',
                        help="Enable detailed torsion info collection")

    parser.add_argument('--torsion_geometric_correction_type', type=str, default="polish_2D",
                        help="Geometric correction type: '2D', 'polish_2D', or '3D'")

    parser.add_argument('--torsion_angular_pxl2deg', type=float, default=0.1,
                        help="Angular pixel-to-degree conversion factor (deg/pxl)")

    parser.add_argument('--torsion_radial_pxl', type=int, default=60,
                        help="Number of radial pixels used for torsion analysis")

    parser.add_argument('--torsion_n_subimgs', type=int, default=100,
                        help="Number of subimages used for stochastic torsion method")

    parser.add_argument('--torsion_max_deg', type=float, default=15.0,
                        help="Maximum torsional rotation in degrees")

    parser.add_argument('--torsion_subimg_angular_deg', type=float, default=90.0,
                        help="Angular size of each subimage window (deg)")

    parser.add_argument('--torsion_subimg_h_pxl', type=int, default=15,
                        help="Subimage height in pixels")

    parser.add_argument('--torsion_coverage_rate_threshold', type=float, default=0.7,
                        help="Threshold for subimage coverage acceptance")

    parser.add_argument('--torsion_force_update_template_interval_sec', type=float, default=5.0,
                        help="Force-update interval for torsion template (not currently used)")

    parser.add_argument('--torsion_TM_algorithm', type=str, choices=["stochastic", "subimage"], default="stochastic",
                        help="Torsion tracking algorithm: 'stochastic' or 'subimage'")

    # --- Visualization ---
    # parser.add_argument('--viz_segmentation', action='store_true',
    #                     help="Visualize segmentation overlay")

    # parser.add_argument('--viz_gaze', action='store_true',
    #                     help="Visualize gaze vector and target")

    parser.add_argument('--viz_results', action='store_true',
                        help="Visualize results (gaze, torsion, etc.)")

    parser.add_argument('--viz_frame_interval', type=int, default=1,
                        help="Frame interval for visualization (e.g., 1 = every frame)")

    parser.add_argument('--alpha_seg_overlay', type=float, default=0.3,
                        
                        help="Alpha transparency for segmentation overlay")
    parser.add_argument('--viz_time_range', type=float, default=5.0,
                        help="Time range (in seconds) for visualization buffer")

    # --- Processing Options ---
    parser.add_argument('--connected_components', action='store_true', default=True,
                        help="Enable connected components filtering in segmentation")

    parser.add_argument('--fit_iris_ellipse', action='store_true', default=True,
                        help="Fit an ellipse to the iris region")

    parser.add_argument('--max_eyeball_param_opt_iters', type=int, default=10000,
                        help="Maximum iterations for eyeball parameter optimization")

    parser.add_argument('--max_frame', type=int, default=argparse.SUPPRESS,
                        help="Maximum number of frames to process (default: all)")

    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for processing frames")

    # --- Thresholds ---
    parser.add_argument('--th_under_exposure', type=float, default=0.1,
                        help="Threshold for underexposed frame filtering")

    parser.add_argument('--th_over_exposure', type=float, default=0.9,
                        help="Threshold for overexposed frame filtering")

    parser.add_argument('--threshold_pupil', type=float, default=0.5,
                        help="Threshold for pupil segmentation")

    parser.add_argument('--threshold_iris', type=float, default=0.5,
                        help="Threshold for iris segmentation")

    parser.add_argument('--threshold_glints', type=float, default=0.5,
                        help="Threshold for glint segmentation")

    parser.add_argument('--threshold_sclera', type=float, default=0.5,
                        help="Threshold for sclera segmentation")

    parser.add_argument('--threshold_confidence_pupil', type=float, default=0.985,
                        help="Confidence threshold for valid pupil detection")

    parser.add_argument('--threshold_confidence_iris', type=float, default=0.985,
                        help="Confidence threshold for valid iris detection")

    parser.add_argument('--blink_threshold', type=float, default=0.735,
                        help="Threshold for detecting blinks")

    # --- Camera Parameters ---
    parser.add_argument('--focal_length', type=float, default=16.0,
                        help="Camera focal length in mm")

    parser.add_argument('--focal_length_pxl', type=float, default=None,
                        help="Optional: focal length in pixels (overrides mm if provided)")

    parser.add_argument('--sensor_size', type=float, nargs=2, default=[4.8, 3.6],
                        help="Sensor size (width height) in mm")
    
    return vars(parser.parse_args())


def main():
    args = make_args()
    # pred_vid = args['pred_vid']
    pred_vid = pred_vid = '/Users/josephzhao/Desktop/Doctoral Project/DeepVOG project/video_test/visualization4TAC2025/ES_gaze_jzhao_v3.mp4'
    args['fit_vid'] = args.get('fit_vid', pred_vid)
    if args['mode'] == 'auto':
        args['mode'] = 'fit'  # or infer intelligently later

    conf_args = get_conf_args(args)
    active_args = {**args, **conf_args}
    pprint(active_args)

if __name__ == "__main__":
    main()