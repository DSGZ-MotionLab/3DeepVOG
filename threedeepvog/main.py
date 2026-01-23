"""
threedeepvog.main

Main entry point for the 3DeepVOG pipeline.

This module orchestrates the full eye-tracking workflow, including:
- video loading and optional automatic downscaling
- segmentation-based eye feature inference
- ellipse fitting
- eyeball model fitting (calibration)
- gaze and torsion estimation
- optional visualization and video output

Modes
-----
- fit:
    Perform eyeball model fitting (calibration) using a calibration video.
    Outputs an eyeball parameter file (JSON).

- predict:
    Perform gaze (and optional torsion) prediction using an existing eyeball model.

- auto / all:
    Run eyeball fitting first (if no existing model is found), then run prediction.

Key Features
------------
- Unified pipeline for fitting and prediction with minimal code duplication.
- Supports both sequential (single-thread) and parallel (multi-thread) execution.
- Automatic video downscaling to a processing resolution (default ~320x240)
  to improve performance while preserving aspect ratio.
- Modular, queue-based architecture for scalability and extensibility.
- Optional outputs:
    - segmentation overlay video
    - fitted model visualization video
    - per-frame gaze and torsion results saved to disk

Notes
-----
- The segmentation model is provided via ModelInference and can be replaced
  by third-party models (e.g., SegResNet, SegFormer).
- Pupil Labs–style 3D gaze estimation (PL algorithm) supports online processing.
- Gaze tracking is enabled in both fit and predict modes; torsion tracking
  is only active in predict mode.
- All geometric parameters (mm2px, focal length, resolution) are automatically
  adjusted after resizing to ensure physical consistency.

TODO
----
- Fully decouple eyeball fitting and gaze prediction into independent runs.
- Improve configuration handling for mixed offline/online workflows.
- Optional real-time streaming input support.

"""

import os, torch, time, cv2, queue, threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

from .args_marker import make_args, get_conf_args
from .module.ModelInference import ModelInference
from .module.PostProcessing import PostProcessing
from .module.EllipseFitting import EllipseFitting
from .module.GazeTracker import GazeTracker
from .module.TorsionTracker import TorsionTracker
from .module.ParamsRender import ParamsRender
from .module.ResultCollector import DiskWriter, OverlayWriter, FitVideoWriter, ResultRouter
# from .module.VideoWriter import FitVideoWriter
# from fast_deepvog3D.validation.SegmentationAnalyzer import SegmentationAnalyzer

def main(args):
    # --- Setup ---
    system_args = get_conf_args(args)
    args = {**args, **system_args}

    # =========================
    # Auto downscale (fastest mode): process+save at ~320x240
    # =========================
    TARGET_W = int(args.get("proc_w", 320))
    TARGET_H = int(args.get("proc_h", 240))
    src_w, src_h = int(args["vid_w_origin"]), int(args["vid_h_origin"])

    if (src_w > TARGET_W) or (src_h > TARGET_H):
        s = min(TARGET_W / src_w, TARGET_H / src_h)  # fit inside target box, keep aspect
        proc_w = int(round(src_w * s))
        proc_h = int(round(src_h * s))
        # snap to even numbers (safer for some codecs)
        proc_w -= proc_w % 2
        proc_h -= proc_h % 2
    else:
        proc_w, proc_h = src_w, src_h

    args["proc_w"], args["proc_h"] = proc_w, proc_h
    args["proc_scale"] = (proc_w / src_w, proc_h / src_h)

    # From here onward: treat processing size as the ONLY size
    args["vid_w"], args["vid_h"] = proc_w, proc_h
    args["resolution"] = (proc_w, proc_h)

    # --- Extract video info ---
    print('\n')
    print(f"**********  START PROCESSING ({args['mode']} mode) **********")
    # print(f"{active_args['mode']} mode...")
    print(f"Video source: {args['input_vid']}")
    print(f"OS: {args['OS']}")
    print(f"Device: {args.get('device', 'cpu')}")
    print(f"Running in {'parallel (multi-thread)' if args.get('is_parallel', False) else 'sequential (single-thread)'} mode")
    print(f"Video info: {args['vid_h_origin']}x{args['vid_w_origin']} -> resize to {args['vid_h']}x{args['vid_w']}, {args['vid_fps']} fps, {args['max_frame']} frames.")
    print('***************************************************\n')

    # Recompute mm2px based on processing resolution
    args["mm2px"] = np.linalg.norm(np.array(args["resolution"])) / np.linalg.norm(np.array(args["sensor_size"]))

    # If focal_length_pxl exists, scale it with width scale (square pixels assumption)
    if args.get("focal_length_pxl") is not None:
        args["focal_length_pxl"] = float(args["focal_length_pxl"]) * args["proc_scale"][0]


    args['mm2px'] = np.linalg.norm(np.array(args['resolution'])) / np.linalg.norm(np.array(args['sensor_size']))
    args['torsion_tracking_flag'] = args.get('torsion_tracking_flag', False) and (args['mode'] == 'predict')
    args['seg_video_flag'] = args.get('seg_video_flag', False) and (args['mode'] == 'predict')
    args['fit_video_flag'] = args.get('fit_video_flag', False) and (args['mode'] == 'predict')

    def make_queues(names):
        return {name: queue.Queue(maxsize=args['batch_size']) for name in names}
    
    Q_SMALL = 4
    Q_VIDEO = 8
    Q_DISK  = 64
    ques = {
        **make_queues(['model_inference', 'post_processing', 'ellipse_fitting']),
        'gaze_tracking': queue.Queue(maxsize=Q_SMALL) if args['gaze_tracking_flag'] else None,
        'torsion_tracking': queue.Queue(maxsize=Q_SMALL) if args['torsion_tracking_flag'] else None,
        'params_rendering': queue.Queue(maxsize=Q_SMALL) if args['fit_video_flag'] else None,

        'ellipse_out': queue.Queue(maxsize=Q_DISK),
        'gaze_out': queue.Queue(maxsize=Q_DISK) if args['gaze_tracking_flag'] else None,
        'torsion_out': queue.Queue(maxsize=Q_DISK) if args['torsion_tracking_flag'] else None,

        'frame_out': queue.Queue(maxsize=Q_VIDEO),
        'segment_out': queue.Queue(maxsize=Q_VIDEO) if args['seg_video_flag'] else None,
        'fitted_frame_out': queue.Queue(maxsize=Q_VIDEO) if args['fit_video_flag'] else None,

        'feedback': queue.Queue(maxsize=1),
    }

    # --- Threads Setup ---
    threads = {'ques': ques, 'tasks': {}}
    tasks = {
        'model_inference': ModelInference(threads, args, daemon=True),
        'post_processing': PostProcessing(threads, args, daemon=True),
        'ellipse_fitting': EllipseFitting(threads, args, daemon=True),
        'gaze_tracker': GazeTracker(threads, args, daemon=True) if args['gaze_tracking_flag'] else None,
        'torsion_tracker': TorsionTracker(threads, args, daemon=True) if args['torsion_tracking_flag'] else None,
        'params_render': ParamsRender(threads, args, daemon=True) if args['fit_video_flag'] else None,
        # 'video_writer': VideoWriter(threads, args, filename_override=args['viz_filename_mp4'], src_que='video_writer', daemon=True) if args['viz_results'] else None,
    }
    threads['tasks'] = tasks

    # ---- Start collector (must start before producers generate outputs) ----
    #TODO: only multi-thread mode
    disk = DiskWriter(out_dir=Path(args["save_folder"]), flush_every=args.get("flush_every", 200), daemon=True)
    disk.start()

    overlay = None
    if args.get("seg_video_flag", False):
        overlay = OverlayWriter(
            out_path=Path(args["save_folder"]) / "seg_overlay.mp4",
            fps=args["vid_fps"],
            size_wh=(args["vid_w"], args["vid_h"]),
            alpha=args.get("seg_overlay_alpha", 0.35),
            daemon=True
        )
        overlay.start()

    fit_writer = None
    if args.get("fit_video_flag", False):
        fit_writer = FitVideoWriter(
            out_path=Path(args["save_folder"]) / "fitted_overlay.mp4",
            fps=args["vid_fps"],
            size_wh=(args["vid_w"], args["vid_h"]),
            daemon=True,
        )
        fit_writer.start()

    router = ResultRouter(threads, args, disk=disk, overlay=overlay, fit_writer=fit_writer, daemon=True)
    router.start()
    threads["collector_router"] = router

    if args['is_parallel']:
        for task in tasks.values():
            if task:
                task.start()

    # --- Analysis Loop ---
    def process_frame_batch(batch, frame_indices):
        frame_batch = {
            'imgs': batch.clone().detach().to(args['device']),
            'idxs': np.arange(frame_indices[0], frame_indices[1] + 1)
        }
        if args['is_parallel']:
            threads['ques']['model_inference'].put(frame_batch)
        else:
            frame_batch = tasks['model_inference'].model_inference(frame_batch)
            frame_batch = tasks['post_processing'].post_processing(frame_batch)
            el_dicts, gaze_batch, torsion_batch = tasks['ellipse_fitting'].ellipse_fitting(frame_batch)
            threads['ques']['ellipse_out'].put(el_dicts)

            if args['seg_video_flag']:
                threads['ques']['segment_out'].put(frame_batch["segs"])

            if args['gaze_tracking_flag']:
                if args['mode'] == 'fit':
                    tasks['gaze_tracker'].batch_fitting(gaze_batch)
                else:
                    gaze_batch = tasks['gaze_tracker'].gaze_tracker(gaze_batch)
                    threads['ques']['gaze_out'].put(gaze_batch)

            if args['torsion_tracking_flag'] and args['mode'] == 'predict':
                if args['gaze_tracking_flag'] and args['torsion_geometric_correction_type']=='3D':
                    torsion_batch['gaze'] = gaze_batch
                torsion_batch = tasks['torsion_tracker'].torsion_tracker(torsion_batch)

                threads['ques']['torsion_out'].put(torsion_batch)
            

    # --- Main Loop ---
    img_batch = torch.zeros((args['batch_size'], args['vid_h'], args['vid_w']), dtype=torch.float32)
    progress_bar = tqdm(total=args['max_frame'], desc="Processing Frames", unit="frame")
    t00 = time.time()

    frame_batch_list = []  # holds BGR frames for overlay video
    for idx in range(args['max_frame']):
        success, frame = args['vid_reader'].read()
        if (not success):
            break

        # ---- resize to processing size (single pipeline) ----
        if (frame.shape[1] != args["vid_w"]) or (frame.shape[0] != args["vid_h"]):
            frame_bgr = cv2.resize(frame, (args["vid_w"], args["vid_h"]), interpolation=cv2.INTER_AREA)
        else:
            frame_bgr = frame
        frame_batch_list.append(frame_bgr)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        img_batch[idx % args["batch_size"]] = torch.from_numpy(gray).float().div_(255.0)
        progress_bar.update(1)

        if (idx + 1) % args['batch_size'] == 0 or idx == args['max_frame'] - 1:
            valid_len = (idx % args['batch_size']) + 1

            batch = img_batch if valid_len == args['batch_size'] else img_batch[:valid_len].clone()
            start = idx + 1 - valid_len

            # ---- send raw frames batch to collector ----
            frames_np = np.stack(frame_batch_list[:valid_len], axis=0)  # (B,H,W,3) uint8
            bid = int(start)  # start index of the batch (unique, monotonic)
            threads['ques']['frame_out'].put((bid, frames_np))

            frame_batch_list.clear()
            process_frame_batch(batch, (start, idx))
            if not(threads['ques']['feedback'].empty()):
                if (threads['ques']['feedback'].get()):
                    print(f"Early stopping at frame {idx + 1}")
                    break

    if args['is_parallel']:
        threads['ques']['model_inference'].put(None)
    else:
        for key in ['ellipse_out', 'segment_out', 'gaze_out', 'torsion_out']:
            if args.get(f'do_{key.split("_")[0]}_tracking', True) or key == 'ellipse_out':
                if threads['ques'][key] != None:
                    threads['ques'][key].put(None)
    progress_bar.close()

    # --- Join Threads and Log ---
    def join_and_log(task_key, label=None):
        task = threads['tasks'].get(task_key)
        if task and args['is_parallel']:
            task.join()
        if label and hasattr(task, 'elapsed_time'):
            fps = args['vid_nr_frames'] / (task.elapsed_time + 1e-6)
            args['elapse_fps'][task_key] = fps
            print(f'Closing thread: {label} (avg/frame: {fps:.2f} fps)')

    args['elapse_fps'] = dict()
    for key in ['model_inference', 'post_processing', 'ellipse_fitting']:
        join_and_log(key, label=key)


    if args['gaze_tracking_flag']:
        # if not params['is_fit']:
        join_and_log('gaze_tracker', label='gaze_tracker')
        # if args['viz_gaze']:
        #     for key in ['gaze_visualization', 'video_writer_gaze']:
        #         join_and_log(key, label=key)

    if args['torsion_tracking_flag']:
        join_and_log('torsion_tracker', label='torsion_tracker')
        tracker = threads['tasks']['torsion_tracker']
        polar_fps = args['vid_nr_frames'] / (getattr(tracker, 'polar_elapsed_time', 1e-6) + 1e-6)
        tm_fps = args['vid_nr_frames'] / (getattr(tracker, 'TM_elapsed_time', 1e-6) + 1e-6)
        print(f'torsion_tracker detail: Polar Transform (avg/frame: {polar_fps:.2f} fps)')
        print(f'torsion_tracker detail: Template Matching (avg/frame: {tm_fps:.2f} fps)')


    # ---- Stop ResultCollector cleanly (parallel mode needs this) ----
    for k in ["frame_out", "ellipse_out", "segment_out", "gaze_out", "torsion_out", "seg_overlay_out", "fitted_frame_out"]:
        q = threads["ques"].get(k)
        if q is not None:
            try:
                q.put_nowait(None)
            except queue.Full:
                q.put(None)
    threads["collector_router"].join()
    disk.join()
    if overlay: overlay.join()
    if fit_writer: fit_writer.join()

    t01 = time.time()
    print(f"Elapsed time total: {t01 - t00:0.2f} sec. (avg/frame: {(t01 - t00) / args['max_frame']:0.3f} sec = {args['max_frame'] / (t01 - t00):0.2f} fps.)")
    args['elapse_fps']['total'] = args['max_frame'] / (t01 - t00)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return threads, args


#camera model, contains info of intrnsic parameters of pupil lab: 
# https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/camera_models.py#L26-L152
def fit_mode(args):
    args['fit_vid'] = args.get('fit_vid', args['pred_vid'])
    args_def = make_args()
    args_def.update(args)
    # model_exists = os.path.exists(args_def.get('eyeball_path', 'xxx'))
    eyeball_path = args_def.get("eyeball_path") or ""
    model_exists = bool(eyeball_path) and os.path.exists(eyeball_path)
    if not model_exists or (args['mode'] == 'all' and input(
        f"Use existing model at {args['eyeball_path']}? (y/n): ").strip().lower() == 'n'):
        args_def['mode'] = 'fit'
        _, fitted_args = main(args_def)
        return fitted_args['eyeball_path']
    else:
        print(f"Using existing eyeball model: {args['eyeball_path']}")
    return {}

def predict_mode(args):
    args['mode'] = 'predict'
    args_def = make_args()
    args_def.update(args)
    return main(args_def)
    
def run_3deepvog(active_args: dict):
    mode = active_args.get('mode', 'auto')

    if mode == 'fit':
        args = fit_mode(args)
        return args

    elif mode == 'predict':
        return predict_mode(args)

    elif mode in ['auto', 'all']:
        print(f"{mode} model")
        active_args['mode'] = mode
        eyeball_path = fit_mode(active_args)
        active_args['eyeball_path'] = eyeball_path
        predict_mode(active_args)
    else:
        raise ValueError(f"Unknown mode 😭: {args['mode']}. Use 'fit', 'predict', 'auto', or 'all'.")
    
    # root = os.path.dirname(active_args['pred_vid'])
    # sample_vid_path = active_args['pred_vid']
    # ellipse_save_path = os.path.join(root, "predict_results", "ellipses.pkl")
    # gaze_save_path = os.path.join(root, "predict_results", "gaze.pkl")
    # torsion_save_path = os.path.join(root, "predict_results", "torsion.pkl")
    # SummaryVideoCreator.summary_video_creator(sample_vid_path, ellipse_save_path, gaze_save_path, torsion_save_path)

if __name__ == '__main__':
    pass
