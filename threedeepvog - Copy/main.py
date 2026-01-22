# Compare to v2, the code separate calibration and prediction using PL algorithm
# -> need to give calibration vid for eyeball fitting and give main vif for prediction

# - model can give third party eye feature segmentation model in ModelInference (default: Berk model)


#Currently, only pupil lab algorithm can achieve online processing whereas legacy code couldn't achieve this.
#TODO
# - Enable to conduct eyeball fitting and gaze/ torsional prediction separetely
# - Enable to conduct eyeball fitting and gaze/ torsional prediction together -> which is the current mode

# all while requiring minimal code changes
# import matplotlib
# matplotlib.use("TkAgg")
# matplotlib.use("Agg")  
from ast import arg
import os, torch, time, cv2, queue, threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
import skvideo.io as skv
from fractions import Fraction
from collections import deque

from .module import SummaryVideoCreator
from .args_marker import make_args, get_conf_args
from .module.ModelInference import ModelInference
from .module.PostProcessing import PostProcessing
from .module.EllipseFitting import EllipseFitting
from .module.GazeTracker_LG import GazeTracker
from .module.TorsionTracker import TorsionTracker
from .module.ResultCollector import ResultCollector
# from .module.VideoWriter import FitVideoWriter
# from fast_deepvog3D.validation.SegmentationAnalyzer import SegmentationAnalyzer

def main(args):
    # --- Setup ---
    system_args = get_conf_args(args)
    args = {**args, **system_args}
    args['mm2px_scaling'] = np.linalg.norm(np.array(args['resolution'])) / np.linalg.norm(np.array(args['sensor_size']))
    args['do_torsion_tracking'] = args.get('do_torsion_tracking', False) and (args['mode'] == 'predict')
    args['write_seg_video'] = args.get('write_seg_video', False) and (args['mode'] == 'predict')
    args['write_fit_video'] = args.get('write_fit_video', False) and (args['mode'] == 'predict')
    def make_queues(names):
        return {name: queue.Queue(maxsize=args['batch_size']) for name in names}

    ques = {
        **make_queues(['model_inference', 'post_processing', 'ellipse_fitting']),
        'gaze_tracking': queue.Queue(maxsize=args['batch_size']) if args['do_gaze_tracking'] else None,
        'torsion_tracking': queue.Queue(maxsize=args['batch_size']) if args['do_torsion_tracking'] else None,
        'write_fit_video': queue.Queue(maxsize=args['batch_size']) if args.get('write_fit_video', False) else None,

        'ellipse_out': queue.Queue(maxsize=args['batch_size'] * 16),
        'gaze_out': queue.Queue(maxsize=args['batch_size'] * 16) if args.get('do_gaze_tracking', False) else None,
        'torsion_out': queue.Queue(maxsize=args['batch_size'] * 16) if args['do_torsion_tracking'] else None,
        'segment_out': queue.Queue(maxsize=8) if args.get('write_seg_video', False) else None,
        'frame_out': queue.Queue(maxsize=8),  # raw frames for overlay video (batched)
        'fitted_frame_out': queue.Queue(maxsize=args['batch_size'] * 16) if args.get('write_fit_video', False) else None,
        # 'visualization': queue.Queue(maxsize=args['batch_size']) if args['viz_results'] else None,
        # 'video_writer': queue.Queue(maxsize=args['batch_size']) if args['viz_results'] else None,
        'feedback': queue.Queue(maxsize=1)
    }

    # --- Threads Setup ---
    threads = {'ques': ques, 'tasks': {}}
    tasks = {
        'model_inference': ModelInference(threads, args, daemon=True),
        'post_processing': PostProcessing(threads, args, daemon=True),
        'ellipse_fitting': EllipseFitting(threads, args, daemon=True),
        'gaze_tracker': GazeTracker(threads, args, daemon=True) if args['do_gaze_tracking'] else None,
        'torsion_tracker': TorsionTracker(threads, args, daemon=True) if args['do_torsion_tracking'] else None,
        # 'fitvideo_writer': FitVideoWriter(threads, args,  daemon=True) if args['write_fit_video'] else None,
        # 'video_writer': VideoWriter(threads, args, filename_override=args['viz_filename_mp4'], src_que='video_writer', daemon=True) if args['viz_results'] else None,
    }
    threads['tasks'] = tasks

    # ---- Start collector (must start before producers generate outputs) ----
    #TODO: only multi-thread mode
    collector = ResultCollector(threads, args, daemon=True)
    collector.start()
    threads["collector"] = collector

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

            if args['write_seg_video']:
                threads['ques']['segment_out'].put(frame_batch["segs"])

            if args['do_gaze_tracking']:
                if args['mode'] == 'fit':
                    tasks['gaze_tracker'].batch_fitting(gaze_batch)
                else:
                    gaze_batch = tasks['gaze_tracker'].gaze_tracker(gaze_batch)
                    threads['ques']['gaze_out'].put(gaze_batch)

            if args['do_torsion_tracking'] and args['mode'] == 'predict':
                if args['do_gaze_tracking'] and args['torsion_geometric_correction_type']=='3D':
                    torsion_batch['gaze'] = gaze_batch
                torsion_batch = tasks['torsion_tracker'].torsion_tracker(torsion_batch)

                threads['ques']['torsion_out'].put(torsion_batch)
            
            # if args['viz_results']:
            #     viz = tasks['visualization'].viz_gaze(frame_batch)
            #     writer = tasks['video_writer']
            #     (writer.write_frame_batch if args['viz_frame_interval'] == 1 else writer.write_single_frame)(viz)


    # --- Main Loop ---
    img_batch = torch.zeros((args['batch_size'], args['vid_h'], args['vid_w']), dtype=torch.float32)
    progress_bar = tqdm(total=args['max_frame'], desc="Processing Frames", unit="frame")
    t00 = time.time()

    frame_batch_list = []  # holds BGR frames for overlay video
    for idx in range(args['max_frame']):
        success, frame = args['vid_reader'].read()
        if (not success):
            break

        # frame from cv2 is usually BGR already; keep it as BGR uint8 for writing
        frame_bgr = frame
        frame_batch_list.append(frame_bgr)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        img_batch[idx % args['batch_size']] = torch.tensor(gray / 255.0, dtype=torch.float32)
        progress_bar.update(1)

        if (idx + 1) % args['batch_size'] == 0 or idx == args['max_frame'] - 1:
            valid_len = (idx % args['batch_size']) + 1

            batch = img_batch if valid_len == args['batch_size'] else img_batch[:valid_len].clone()
            start = idx + 1 - valid_len

            # ---- send raw frames batch to collector ----
            frames_np = np.stack(frame_batch_list[:valid_len], axis=0)  # (B,H,W,3) uint8
            threads['ques']['frame_out'].put(frames_np)
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


    if args['do_gaze_tracking']:
        # if not params['is_fit']:
        join_and_log('gaze_tracker', label='gaze_tracker')
        # if args['viz_gaze']:
        #     for key in ['gaze_visualization', 'video_writer_gaze']:
        #         join_and_log(key, label=key)

    if args['do_torsion_tracking']:
        join_and_log('torsion_tracker', label='torsion_tracker')
        tracker = threads['tasks']['torsion_tracker']
        polar_fps = args['vid_nr_frames'] / (getattr(tracker, 'polar_elapsed_time', 1e-6) + 1e-6)
        tm_fps = args['vid_nr_frames'] / (getattr(tracker, 'TM_elapsed_time', 1e-6) + 1e-6)
        print(f'torsion_tracker detail: Polar Transform (avg/frame: {polar_fps:.2f} fps)')
        print(f'torsion_tracker detail: Template Matching (avg/frame: {tm_fps:.2f} fps)')
        # if args['viz_torsion']:
        #     for key in ['torsion_visualization', 'video_writer_torsion']:
        #         join_and_log(key, label=key)

    # ---- Stop ResultCollector cleanly (parallel mode needs this) ----
    for k in ["frame_out", "ellipse_out", "segment_out", "gaze_out", "torsion_out", "fitted_frame_out"]:
        q = threads["ques"].get(k)
        if q is not None:
            try:
                q.put_nowait(None)
            except queue.Full:
                q.put(None)
    threads["collector"].join()

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
