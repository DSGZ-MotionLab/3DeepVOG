# Compare to v2, the code separate calibration and prediction using PL algorithm
# -> need to give calibration vid for eyeball fitting and give main vif for prediction

# - model can give third party eye feature segmentation model in ModelInference (default: Berk model)


#Currently, only pupil lab algorithm can achieve online processing whereas legacy code couldn't achieve this.
#TODO
# - Enable to conduct eyeball fitting and gaze/ torsional prediction separetely
# - Enable to conduct eyeball fitting and gaze/ torsional prediction together -> which is the current mode
# import sys    # sys.path.append("D:/git/DeepVOG3DTorch/DeepVOG/deepvog3D")

# torch.compile is the latest method to speed up your PyTorch code! 
# torch.compile makes PyTorch code run faster by JIT-compiling PyTorch code into optimized kernels, 
# all while requiring minimal code changes
import matplotlib
# matplotlib.use("TkAgg")
# matplotlib.use("Agg")  

import os, torch, time, cv2, queue
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.read_and_save import get_video_info_torch
# from utils.default_setting import default_params
from .module.ModelInference import ModelInference
from .module.PostProcessing import PostProcessing
from .module.EllipseFitting import EllipseFitting
from .module.GazeTracker_PL import GazeTracker   #able to run GazeTracker_LG, GazeTracker, GazeTracker_dev3
from .module.TorsionTracker import TorsionTracker

# from fast_deepvog3D.VisualizerResults import VisualizerResults
# from fast_deepvog3D.VideoWriter import VideoWriter
# from fast_deepvog3D.validation.SegmentationAnalyzer import SegmentationAnalyzer

from args_marker_v2 import get_conf_args, make_args


def collect_from_queue(queue, process_fn=None, stack=False):
    results = []
    while True:
        item = queue.get()
        if item is None:
            break
        if process_fn:
            item = process_fn(item)
        results.append(item)
    if stack:
        return torch.stack(results)
    return results

def collect_results(threads, args):
    df_ellipses, df_gaze_out, segment_map_all, torsion_out,\
    current_iris_imgs_out, template_iris_imgs_out = None, None, None, None, None, None
    # ---- Ellipses ----
    df_ellipses = pd.concat(
        collect_from_queue(
            threads['ques']['ellipse_out'],
            process_fn=lambda d: pd.DataFrame({k: v.cpu().numpy() for k, v in d.items() if isinstance(v, torch.Tensor)})
        ),
        axis=0, ignore_index=True
    )
    # ---- Segment Map ----
    if args.get('extract_segment_map', False):
        segment_map = collect_from_queue(threads['ques']['segment_out'])
        segment_map_all = torch.cat(segment_map).cpu().numpy()

    # ---- Torsion Tracking ----
    if args['do_torsion_tracking']:
        torsion_data = collect_from_queue(threads['ques']['torsion_out'])
        torsion_out, current_imgs, current_imgs, templates = [], [], [], []
        for t in torsion_data:
            # if args['viz_torsion'] or args['torsion_collecte_detail']:
            if args['torsion_collecte_detail']:
                torsion_out.extend(t['torsion_angles'])
                current_imgs.append(t['current_iris_imgs'])
                templates.append(t['template_iris_imgs'])
            else:
                torsion_out.extend(t)
        torsion_out = torch.tensor(torsion_out).cpu().numpy()
        # if args['viz_torsion'] or args['torsion_collecte_detail']:
        if args['torsion_collecte_detail']:
            current_iris_imgs_out = torch.cat(current_imgs).cpu().numpy()
            current_iris_imgs_out = torch.cat(current_imgs).cpu().numpy()
            template_iris_imgs_out = torch.stack(templates).cpu().numpy()
        
    # ---- Gaze Tracking ----
    df_gaze_out = pd.concat(
        collect_from_queue(threads['ques']['gaze_out'], process_fn=lambda d: pd.DataFrame(d)),
        axis=0, ignore_index=True
    ) if args['do_gaze_tracking'] else pd.DataFrame()
    return df_ellipses, df_gaze_out, segment_map_all, torsion_out, current_iris_imgs_out, template_iris_imgs_out


def main(args):
    # --- Setup ---
    system_args = get_conf_args(args)
    args = {**args, **system_args}
    args['mm2px_scaling'] = np.linalg.norm(np.array(args['resolution'])) / np.linalg.norm(np.array(args['sensor_size']))

    def make_queues(names):
        return {name: queue.Queue(maxsize=args['batch_size']) for name in names}

    ques = {
        **make_queues(['model_inference', 'post_processing', 'ellipse_fitting']),
        'gaze_tracking': queue.Queue(maxsize=args['batch_size']) if args['do_gaze_tracking'] else None,
        'torsion_tracking': queue.Queue(maxsize=args['batch_size']) if args['do_torsion_tracking'] else None,

        'ellipse_out': queue.Queue(),
        'gaze_out': queue.Queue() if args.get('do_gaze_tracking', False) else None,
        'torsion_out': queue.Queue() if args.get('do_torsion_tracking',False) else None,
        'segment_out': queue.Queue() if args.get('extract_segment_map', False) else None,

        # 'ellipse_visualization': queue.Queue(maxsize=args['batch_size']) if args['viz_segmentation'] else None,
        # 'video_writer_ellipses': queue.Queue(maxsize=args['batch_size']) if args['viz_segmentation'] else None,
        # 'gaze_visualization': queue.Queue(maxsize=args['batch_size']) if args['viz_gaze'] else None,
        # 'video_writer_gaze': queue.Queue(maxsize=args['batch_size']) if args['viz_gaze'] else None,
        'visualization': queue.Queue(maxsize=args['batch_size']) if args['viz_results'] else None,
        'video_writer': queue.Queue(maxsize=args['batch_size']) if args['viz_results'] else None,
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

        'visualization': VisualizerResults(threads, args, daemon=True) if args['viz_results'] else None,
        'video_writer': VideoWriter(threads, args, filename_override=args['viz_filename_mp4'], src_que='video_writer', daemon=True) if args['viz_results'] else None,
    }
    threads['tasks'] = tasks

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

            if args['extract_segment_map']:
                threads['ques']['segment_out'].put(frame_batch)

            if args['do_gaze_tracking']:
                if args['mode'] == 'fit':
                    tasks['gaze_tracker'].batch_fitting(gaze_batch)
                else:
                    gaze_batch = tasks['gaze_tracker'].gaze_tracker(gaze_batch)
                    threads['ques']['gaze_out'].put(gaze_batch)

            if args['do_torsion_tracking']:
                if args['do_gaze_tracking'] and args['torsion_geometric_correction_type']=='3D':
                    torsion_batch['gaze'] = gaze_batch
                torsion_batch = tasks['torsion_tracker'].torsion_tracker(torsion_batch)
                # if args['viz_torsion']:
                #     viz = tasks['torsion_visualization'].viz_torsion(torsion_batch)
                #     writer = tasks['video_writer_torsion']
                #     (writer.write_frame_batch if args['viz_frame_interval'] == 1 else writer.write_single_frame)(viz)
                #     threads['ques']['torsion_out'].put(torsion_batch['torsion_angles'])
                # else:
                threads['ques']['torsion_out'].put(torsion_batch)
            
            if args['viz_results']:
                viz = tasks['visualization'].viz_gaze(frame_batch)
                writer = tasks['video_writer']
                (writer.write_frame_batch if args['viz_frame_interval'] == 1 else writer.write_single_frame)(viz)


    # --- Main Loop ---
    img_batch = torch.zeros((args['batch_size'], args['vid_h'], args['vid_w']), dtype=torch.float32)
    progress_bar = tqdm(total=args['max_frame'], desc="Processing Frames", unit="frame")
    t00 = time.time()

    for idx in range(args['max_frame']):
        success, frame = args['vid_reader'].read()
        if (not success):
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img_batch[idx % args['batch_size']] = torch.tensor(gray / 255.0, dtype=torch.float32)
        progress_bar.update(1)
        if (idx + 1) % args['batch_size'] == 0 or idx == args['max_frame'] - 1:
            valid_len = (idx % args['batch_size']) + 1
            batch = img_batch if valid_len == args['batch_size'] else img_batch[:valid_len].clone()
            start = idx + 1 - valid_len
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

    # if args['viz_segmentation']:
    #     for key in ['ellipse_visualization', 'video_writer_ellipses']:
    #         join_and_log(key, label=key)

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

    t01 = time.time()
    print(f"Elapsed time total: {t01 - t00:0.2f} sec. (avg/frame: {(t01 - t00) / args['max_frame']:0.3f} sec = {args['max_frame'] / (t01 - t00):0.2f} fps.)")
    args['elapse_fps']['total'] = args['max_frame'] / (t01 - t00)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return threads, args


#camera model, contains info of intrnsic parameters of pupil lab: 
# https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/camera_models.py#L26-L152
def run_3deepvog(active_args):
    active_args['fit_vid'] = active_args.get('fit_vid', active_args['pred_vid'])
    mode = active_args.get('mode', 'auto')
    def fit_model():
        active_args['mode'] = 'fit'
        _, fitted_args = main(active_args)
        active_args['eyeball_path'] = fitted_args['eyeball_path']
        return fitted_args

    def predict_model():
        active_args['mode'] = 'predict'
        return main(active_args)

    if active_args['mode'] == 'fit':
        args = fit_model()
        return None, args

    elif active_args['mode'] == 'predict':
        return predict_model()

    elif active_args['mode'] in ['auto', 'all']:
        print(f"{active_args['mode']} model")
        model_exists = os.path.exists(active_args.get('eyeball_path', ''))
        if not model_exists or (mode == 'all' and input(
            f"Use existing model at {active_args['eyeball_path']}? (y/n): ").strip().lower() == 'n'):
            args = fit_model()
        else:
            print(f"Using existing eyeball model: {active_args['eyeball_path']}")
            # Optional: assign again if needed
            active_args['eyeball_path'] = active_args['eyeball_path']
        return predict_model()
    else:
        raise ValueError(f"Unknown mode 😭: {active_args['mode']}. Use 'fit', 'predict', 'auto', or 'all'.")


if __name__ == '__main__':
    # import debugpy
    # debugpy.listen(5678)
    # print("Waiting for debugger attach")
    # debugpy.wait_for_client()
    # print("Debugger attached - starting program")
    
    active_args = make_args()
    # from fast_deepvog3D.model3D.segmentation_model import AttentionUnet_3in3out_model, Unet_3in4out_model, SegResNet_3in3out_model
    # model_weight_path = r"D:\jzhao\DeepVOG-project\result\segmentation\trained_model\2025-01-18_AttentionUnet_3in3out\final_best_model.pth"
    active_args.update({
        # 'fit_vid': XXXX,   #comment out: use pred_vid for fitting eyeball model
        # 'pred_vid': r"D:\jzhao\DeepVOG-project\datasets\Patientrecording\sub3\trial.avi",
        'pred_vid': r"D:\jzhao\DeepVOG-project\video_test\test_cornea_el_fitting\ES_gaze_jzhao_v3.mp4",
        'is_parallel': True,   #True: multithread, False: single thread (sequential)
        'batch_size': 32,   #batch size for processing frames
        'device': 'cuda',   #comment out: use default device, e.g. 'cuda', 'mps', 'cpu'
        'segmentation_model': None,   #TODO: need to cahnge: comment out or give a model name, e.g. 'SegResNet_3in3out'
        'segmentation_model_weights_path': None,  #TODO: need to cahnge: comment out or give a model name, e.g. 'SegResNet_3in3out'
        # 'max_frame': 1000,  # comment out: default is all frames
        # 'eyeball_path': xxx,   #can define custom eyeball model path, comment out if not needed
        'extract_segment_map': False,  #"all", "sclera", "False"
        'do_gaze_tracking': True,
        'eyeball_model': 'simple', #simple or LeGrand or PL
        'do_torsion_tracking': False,
        'torsion_collecte_detail': False,
        'focal_length': 16.0,  # mm
        'sensor_size': (4.8, 3.6),  # mm
        'viz_segmentation': False,
        'viz_gaze': False,
        'viz_torsion': False
    })

    # if test_flag:
    # active_args['pred_vid'] = r"D:\jzhao\DeepVOG-project\video_test\test_cornea_el_fitting\ES_gaze_jzhao_v3.mp4"
    threads, args = run_3deepvog(active_args = active_args,
                                    mode = 'auto'   #fit / predict / auto / all
                                    )
    # save params as pickle file
    df_ellipses, df_gaze_out, sclera_roi_all, torsion_out,\
    current_iris_imgs_out, template_iris_imgs_out = collect_results(threads, args)

    # else:
    #     # model_types = ['simple', 'PL', 'LeGrand']
    #     # calib_types = ["calibME", "freelook", "narrowranged"]
    #     focal_lengths = [16.0]
    #     subjects = ['p10121']
    #     calib_types = ['calibME'] 
    #     calib_ix = ["001"]
    #     model_types = ['PL']  
    #     data_root = r"D:\\jzhao\\DeepVOG-project\\datasets\\Gaze_EyeSeeCam\\datasetC\\processed_data"
    #     seg_save_root = r"D:\\jzhao\\DeepVOG-project\\result\\segmentation\\DatasetC"
    #     for fcl in focal_lengths:
    #         gaze_save_root = rf"D:\\jzhao\\DeepVOG-project\\result\\gaze_estimation\\DatasetC\\fcl{str(int(fcl))}_ransacNew"
    #         active_args['focal_length'] = fcl
    #         for sub in subjects:
    #             for ix in calib_ix:
    #                 for calib_type in calib_types:
    #                     calib_trial = f"{ix}_{calib_type}"
    #                     if save_flag:
    #                         os.makedirs(seg_save_root, exist_ok=True)
    #                         os.makedirs(gaze_save_root, exist_ok=True)
    #                     # calib_vid = os.path.join(data_root, sub, f'{calib_trial}.mp4')
    #                     calib_vid = os.path.join(data_root, sub, f'{calib_trial}.mp4')
    #                     # calib_main_vid = os.path.join(data_root, sub, f'{calib_trial}-trial.mp4')
    #                     for eyeball_model in model_types:
    #                         active_args["eyeball_model"] = eyeball_model
    #                         # eyeball_save_path = os.path.join(gaze_save_root, f'{sub}_{calib_trial}_EyeModel_{set_param["eyeball_model"]}.json')
    #                         gaze_save_path = os.path.join(gaze_save_root, f'{sub}_{calib_trial}_gaze_{active_args["eyeball_model"]}.csv')
    #                         ellipse_save_path = os.path.join(seg_save_root, f'{sub}_{calib_trial}_ellipse.csv')
    #                         # torsion_save_path = os.path.join(gaze_save_root, f'{sub}_{calib_trial}_torsion_{set_param["eyeball_model"]}.csv')
    #                         params_save_path = os.path.join(gaze_save_root, f'{sub}_{calib_trial}_params_{active_args["eyeball_model"]}.json')
    #                         ff_vid = os.path.join(data_root, sub, f'{calib_trial}-trial.mp4')

    #                         active_args['pred_vid'] = ff_vid
    #                         active_args['fit_vid'] = calib_vid
    #                         threads, args = run_3deepvog(active_args = active_args,
    #                             mode = 'all'   #fit / predict / auto / all
    #                             )
    #                         # save params as pickle file
    #                         df_ellipses, df_gaze_out, sclera_roi_all, torsion_out,\
    #                         current_iris_imgs_out, template_iris_imgs_out = collect_results(threads, args)
    #                         if save_flag:
    #                             df_ellipses.to_csv(ellipse_save_path, index=False)
    #                             df_gaze_out.to_csv(gaze_save_path, index=False)


    # # ---- Visualization ----
    # # The assumed physiological bounds:   https://pupil-labs.com/releases/core-v3-4
    # # slightly different from definition in detector_3d
    # # Phi and theta ranges are relative to the eye camera's optical axis. 
    # # The eye ball center ranges are defined relative to the origin of the eye camera's 3d coordinate system.
    # # The model_confidence will be set to 0.0 if the gaze direction cannot be calculated.
    # # confidence_mask = df_gaze_out.model_confidence == 1
    # if not(df_gaze_out.empty) and not(test_flag):
    #     non_interest_ix = df_gaze_out.confidence.values < args['threshold_confidence_pupil']
    #     from utils.gaze_process import gaze_extract_GT, gaze_extract_LG, gaze_extract_PL
    #     if save_flag:
    #         df_gaze_out.to_csv(gaze_save_path, index=False)
    #     _, _, _, (max_calb, h, w, channels), _, _, vid_fps = get_video_info_torch(calib_vid)
    #     if active_args['eyeball_model'] == 'PL':
    #         gaze_x, gaze_y = gaze_extract_PL(df_gaze_out[max_calb:], thr_confidence = 0.96)
    #     else:
    #         gaze_x, gaze_y = gaze_extract_LG(df_gaze_out[max_calb:], thr_confidence = 0.96)
    #     gaze_GT_path = os.path.join(data_root, sub, 'trial_GT.csv')
    #     df_GT_gaze_out = pd.read_csv(gaze_GT_path)
    #     gaze_x_GT, gaze_y_GT = gaze_extract_GT(df_GT_gaze_out)
    #     plt.figure(figsize=(15, 5))
    #     plt.subplot(2, 1, 1)
    #     plt.plot(gaze_x, label = "Estimated", color='orange', linewidth = 0.5)
    #     plt.plot(gaze_x_GT, label = "Ground Truth", color='red', linewidth = 0.5)
    #     # plt.ylim([-30, 30])
    #     plt.ylabel('[°]')
    #     plt.legend()
    #     plt.title('Horizontal Gaze')

    #     plt.subplot(2, 1, 2)
    #     plt.plot(gaze_y, label = "Estimated", color='orange', linewidth = 0.5)
    #     plt.plot(-gaze_y_GT, label = "Ground Truth", color='red', linewidth = 0.5)
    #     plt.ylim([-30, 30])
    #     plt.ylabel('[°]')
    #     plt.legend()
    #     plt.title('Vertical Gaze')
    #     plt.tight_layout()
    #     plt.show()

    # plt.figure(figsize=(15, 5))
    # plt.plot(torsion_out, linewidth = 0.25, label = "torsion")
    # plt.show()

