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

import os
from threedeepvog.main import run_3deepvog
import matplotlib.pyplot as plt

# import debugpy
# debugpy.listen(5678)
# print("Waiting for debugger attach")
# debugpy.wait_for_client()
# print("Debugger attached - starting program")
if __name__ == '__main__':
    args = {
        # 'fit_vid': XXXX,   #comment out: use pred_vid for fitting eyeball model
        # 'pred_vid': r"D:\jzhao\DeepVOG-project\datasets\Patientrecording\sub3\trial.avi",
        'pred_vid': r"D:\jzhao\DeepVOG-project\video_test\test_cornea_el_fitting\ES_gaze_jzhao_v3.mp4",
        # 'pred_vid': r"D:\jzhao\DeepVOG-project\video_test\segmentation_fMRI_VOG\test_fMRI_VOG.mp4",
        # 'pred_vid': r"D:\jzhao\DeepVOG-project\video_test\TEyeD\GW_21_5\GW_21_5.mp4",
        # 'pred_vid': r"D:\jzhao\DeepVOG-project\video_test\from_ZUMUTU\MVS_SMP_test-Left.avi",
        # 'pred_vid': r"D:\jzhao\DeepVOG-project\datasets\Pupil_Segmentation\TEyeD\Dikablis\VIDEOS\DikablisSA_16_3.mp4",
        # 'is_parallel': True,   #True: multithread, False: single thread (sequential)
        # 'batch_size': 32,   #batch size for processing frames
        'device': 'cuda',   #comment out: use default device, e.g. 'cuda', 'mps', 'cpu'
        'model_frozen': True,  
        # 'segmentation_model': xxx,   #e.g. 'SegResNet_3in3out'
        # 'segmentation_model_weights_path': xxx,  #TODO:  e.g. 'SegResNet_3in3out'
        # # 'max_frame': 1000,  # comment out: default is all frames
        'eyeball_model': 'PL', #simple or LeGrand or PL
        # 'extract_segment_map': False,  #"all", "sclera", "False"

        'connected_components': 'morph',   #False / None / 'morph' / 'largest'
        # 'torsion_collecte_detail': False,
        'focal_length': 16.0,  # mm
        # 'sensor_size': (4.8, 3.6),  # mm
        'gaze_tracking_flag': True,
        'torsion_tracking_flag': True,
        'seg_video_flag': True,
        'fit_video_flag': True,
        'write_seg_video_type': 'processed',   # 'processed', 'raw'
        # 'log_dir':  r"D:\jzhao\DeepVOG-project\video_test\segmentation_fMRI_VOG\log",  # None
        'eyeball_path': None,   #can define custom eyeball model path, comment out if not needed
        'mode': 'auto'   #fit / predict / auto / all
    }

    run_3deepvog(active_args = args)
    


    # df_ellipses, df_gaze_out, _, torsion_out, *_ = collect_results(threads, updated_params)
    # base_name = os.path.splitext(os.path.basename(args['pred_vid']))[0]
    # df_ellipses.to_csv(f"{base_name}_ellipses.csv", index=False)
    # df_gaze_out.to_csv(f"{base_name}_gaze.csv", index=False)

        
        # # save params as pickle file
        # df_ellipses, df_gaze_out, sclera_roi_all, torsion_out,\
        # current_iris_imgs_out, template_iris_imgs_out = collect_results(threads, args)

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

