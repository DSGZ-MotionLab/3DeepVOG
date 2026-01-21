import ast
import os

import torch

from ..tool.cart2sph import cart2sph_batch_PL
os.environ['VISPY_APP'] = 'pyside6'
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import pickle
from matplotlib import pyplot as plt
# from fast_deepvog3D.model3D.segmentation_model import SegResNet_3in3out_model
from scipy.spatial.transform import Rotation as Quaternion_Rotation
# from deepvog_processer import deepvog_processer_v2
from ..utils.read_and_save import get_video_info_cv2
from ..utils.gaze_process import opt_transform_v3
from ..utils.torsion_process import clean_signal, interpolate_nan
from ..utils.visualization import gen_sphere_mesh, fit_legrand_model, \
SegFitVisualizer, VispyEyeballRenderer, EyeMovementVisualizer, stack_polar_maps_colored
from vispy import scene
from vispy.visuals.transforms import STTransform
from vispy.visuals.filters.clipping_planes import PlanesClipper
from vispy.scene.cameras import TurntableCamera
from numpy.lib.stride_tricks import sliding_window_view


def cart2sph_batch_PL_np(N, eps=1e-8):
    norm = np.linalg.norm(N, axis=0, keepdims=True)
    N = N / np.maximum(norm, eps)
    phi = np.arctan2(N[2], N[0]) + np.pi / 2
    theta = np.arccos(np.clip(N[1], -1.0, 1.0)) - np.pi / 2
    return np.rad2deg(phi), np.rad2deg(theta)

    # --- Params ---
def summary_video_creator(sample_vid, ellipse_save_path, gaze_save_path, torsion_save_path):
    fcl = 16.0
    # root = r"D:\jzhao\DeepVOG-project\video_test\segmentation_fMRI_VOG"
    # sample_vid = os.path.join(root, "test_fMRI_VOG.mp4")
    # fitted_overlay_vid = os.path.join(root, "predict_results", "fitted_overlay.mp4")
    # seg_overlay_vid = os.path.join(root, "predict_results", "seg_overlay.mp4")
    eyeball_path = sample_vid.replace(".mp4", "_PL_eyeball_model.json")
    df_eyeball = pd.read_json(eyeball_path, orient='index')
    # df_gaze_GT = pd.read_csv(r"D:\jzhao\DeepVOG-project\datasets\Patientrecording\sub3\trial.csv")
    df_gaze_GT = None
    # --- Data ---
    print("Loading data...")
    # output_pkl_path = os.path.join(root_path, "deepvog_outputs.pkl")
    # with open(output_pkl_path, 'rb') as f:
    #     data = pickle.load(f)
    # ellipse_save_path = os.path.join(root, "predict_results", "ellipse.csv")
    # gaze_save_path = os.path.join(root, "predict_results", "gaze.csv")
    # torsion_save_path = os.path.join(root, "predict_results", "torsion.csv")
    df_ellipses = pd.read_pickle(ellipse_save_path)
    df_gaze_out = pd.read_pickle(gaze_save_path)
    torsion_out = pd.read_pickle(torsion_save_path)
    # df_ellipses = data["df_ellipses"]
    # df_gaze_out = data["df_gaze_out"]
    # sclera_roi_all = data["sclera_roi_all"]
    # torsion_out = data["torsion_out"]
    # current_iris_imgs_out = data["current_iris_imgs_out"]
    # template_iris_imgs_out = data["template_iris_imgs_out"]
    # params = data["params"]
    vid_w, vid_h, fps, n_frame = get_video_info_cv2(sample_vid)
    root_path = os.path.dirname(sample_vid)

    camera_params = {
        'fcl_mm': fcl, 'resolution': (vid_w, vid_h),
        'mm2px_scaling': np.linalg.norm(np.array([vid_w, vid_h]) / np.linalg.norm([4.8, 3.6])),
        'sensor_size': (4.8, 3.6), 'fps': fps,
    }


    if df_gaze_GT is not None:
        GT_x, GT_y = -np.rad2deg(df_gaze_GT.hor), -np.rad2deg(df_gaze_GT.ver)
        GT_x -= np.nanmean(GT_x); GT_y -= np.nanmean(GT_y)
        df_gaze_out = df_gaze_out.loc[:len(GT_x)-1, :]
        normals = np.array([row['normal'] for row in df_gaze_out['circle_3d']]).T
        conf = df_gaze_out['confidence'].values

        GT_list = [GT_x, GT_y]
        final_x, final_y, opt_M = opt_transform_v3(GT_list, normals, conf)
    else:
        normals = np.array([row['normal'] for row in df_gaze_out['circle_3d']]).T
        final_x, final_y = cart2sph_batch_PL_np(normals)

    # clean_torsion = clean_signal(torsion_out, fps=fps, drift_removal=True, drift_window=20, lowcut=2, highcut=20.0, order=4)
    torsion_out = np.asarray(torsion_out).squeeze()
    clean_torsion = clean_signal(torsion_out, fps=fps, drift_removal=True, drift_window=20, lowcut=2, highcut=20.0, order=4)
    processed_signal = {"horizontal": final_x, "vertical": final_y, "torsion": clean_torsion}
    processed_torsion = np.deg2rad(clean_torsion)

    # --- Eyeball params ---
    eyeball_center = df_eyeball.iloc[0:3, :].values.T[0]
    eyeball_params = {
        'eyeball_radius': 12, 'corneaball_radius': 7.8, 'limbus_radius': 6.0,
        'dp': np.sqrt(12**2 - 6.0**2), 'num_grids': 25,
        'line_thickness': max(1, int(np.round((np.linalg.norm(camera_params['resolution'])/500)))),
        'eyeball_center': eyeball_center,
    }


    # Example usage
    # combined_img = stack_polar_maps(polar_current, polar_TM)
    # --- Output setup ---
    r_e, r_c, r_s = eyeball_params['eyeball_radius'], eyeball_params['corneaball_radius'], eyeball_params['limbus_radius']
    d_p, d_lim_cornea = np.sqrt(r_e**2 - r_s**2), np.sqrt(r_c**2 - r_s**2)
    limbus_theta_eye, limbus_theta_cornea = np.arccos(d_p / r_e), np.arccos(d_lim_cornea / r_c)

    large_meshes = gen_sphere_mesh(r_e, eyeball_params['num_grids'], insert_theta=limbus_theta_eye)
    small_meshes = gen_sphere_mesh(r_c, eyeball_params['num_grids'], insert_theta=limbus_theta_cornea)


    # --- Load frames ---
    frames = []
    cap = cv2.VideoCapture(sample_vid)
    for _ in range(n_frame):
        ret, frame = cap.read()
        if not ret: break
        frames.append(frame)
    # Release capture and stack frames
    cap.release()
    frames = np.stack(frames)

    frame_size = frames.shape[1:3]

    C_pannel_w = 300
    C_pannel_h = 300
    R_pannel_h = 300
    # C_pannel_h = 300
    R_pannel_w = frame_size[1] + C_pannel_w
    # Compute vertical placement
    L_start_y = (C_pannel_h - frame_size[0]) // 2
    L_end_y = L_start_y + frame_size[0]

    vispy_renderer = VispyEyeballRenderer(eyeball_params['eyeball_center'],
                                        width = C_pannel_w,
                                        height = C_pannel_h
                                        )

    # seg_fit_renderer = SegFitVisualizer(height= h_pannel, 
    #                                     frame_size=frame_size,
    #                                     camera_params=camera_params,
    #                                     eyeball_params=eyeball_params
    #                                     )

    eye_move_vis = EyeMovementVisualizer(
                                        processed_signal=processed_signal,
                                        width = R_pannel_w,
                                        height= R_pannel_h,
                                        max_frame_show=500
                                    )


    output_path = os.path.join(root_path, "gaze_combined_with_3Deye_v2.mp4")
    writer = cv2.VideoWriter(output_path, 
                            cv2.VideoWriter_fourcc(*"mp4v"), 
                            int(round(camera_params['fps'])), 
                            (R_pannel_w, C_pannel_h + R_pannel_h)) # (w_total, h)


    # n_frame = 2000
    # n_frame = len(frames)-1
    n_frame = min(len(frames), len(df_ellipses), len(df_gaze_out), len(processed_torsion))
    fit_model_times = []
    seg_fit_renderer_times = []
    vispy_renderer_times = []
    eye_move_vis_times = []


    for tp in tqdm(range(n_frame), desc="Rendering video"):
        frame = frames[tp]
        el_info = df_ellipses.iloc[tp]
        # sclera_mask = sclera_roi_all[tp]
        # polar_current = current_iris_imgs_out[tp]
        # polar_TM = template_iris_imgs_out[tp]
        # if np.any(np.isnan(el_info[["iris_center_x", "pupil_center_x"]].values.astype(float))):
        #     continue
        def has_nan(v):
            try:
                a = np.asarray(v, dtype=float)
                return np.any(np.isnan(a))
            except Exception:
                return True  # treat malformed values as invalid

        for tp, el_info in df_ellipses.iterrows():
            if has_nan(el_info["iris_center_x"]) or has_nan(el_info["pupil_center_x"]):
                continue

        # --- Timing: model fitting ---
        t0 = time.perf_counter()
        result = fit_legrand_model(eyeball_params, df_gaze_out.iloc[tp], processed_torsion[tp],
                                large_meshes, small_meshes, eyeball_params['num_grids'], use_mask=True)
        t_fit = time.perf_counter() - t0
        fit_model_times.append(t_fit)

        # if result is None:
        #     continue

        # polar_panel = stack_polar_maps_colored(polar_current, polar_TM, output_size=(w_pannel, h_pannel - C_pannel_h))
        # --- Timing: vispy rendering ---
        t0 = time.perf_counter()
        center_panel = vispy_renderer.render(result)
        t_vispy = time.perf_counter() - t0
        vispy_renderer_times.append(t_vispy)

        # --- Timing: segmentation + overlay rendering ---
        t0 = time.perf_counter()

        left_panel = np.full((C_pannel_h, frame_size[1], 3), 255, dtype=np.uint8)

        h, w = frame.shape[:2]
        scale = min(C_pannel_h / h, frame_size[1] / w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        frame_rs = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        y0 = (C_pannel_h - new_h) // 2
        x0 = (frame_size[1] - new_w) // 2
        left_panel[y0:y0 + new_h, x0:x0 + new_w, :] = frame_rs

        combined_up = np.hstack((left_panel, center_panel))

        t_seg = time.perf_counter() - t0
        seg_fit_renderer_times.append(t_seg)

        # --- Timing: signal plotting ---
        t0 = time.perf_counter()
        right_panel = eye_move_vis.render(tp)
        t_plot = time.perf_counter() - t0
        eye_move_vis_times.append(t_plot)

        combined = np.vstack([combined_up, right_panel])
        writer.write(combined.astype(np.uint8))

    writer.release()
    print(f"Video saved to: {output_path}")

    # --- FPS reporting ---
    def compute_fps(times):
        times = np.array(times)
        return 1.0 / times.mean()

    print("\nAverage FPS per component:")
    print(f"  Model fitting     : {compute_fps(fit_model_times):.2f} fps")
    print(f"  SegFit rendering  : {compute_fps(seg_fit_renderer_times):.2f} fps")
    print(f"  VisPy rendering   : {compute_fps(vispy_renderer_times):.2f} fps")
    print(f"  Signal plotting   : {compute_fps(eye_move_vis_times):.2f} fps")



    # # Assuming all your `frames`, `df_ellipses`, `sclera_roi_all`, etc. are preloaded
    # from concurrent.futures import ThreadPoolExecutor
    # executor = ThreadPoolExecutor(max_workers=2)  # For seg_fit + eye_plot

    # fit_model_times = []
    # seg_fit_renderer_times = []
    # vispy_renderer_times = []
    # eye_move_vis_times = []
    # for tp in tqdm(range(n_frame), desc="Rendering video"):
    #     frame = frames[tp]
    #     el_info = df_ellipses.iloc[tp]
    #     sclera_mask = sclera_roi_all[tp]
    #     if np.any(np.isnan(el_info[["iris_center_x", "pupil_center_x"]])):
    #         continue

    #     # --- Model fitting ---
    #     t0 = time.perf_counter()
    #     result = fit_legrand_model(
    #         eyeball_params, df_gaze_out.iloc[tp], processed_torsion[tp],
    #         large_meshes, small_meshes, eyeball_params['num_grids'], use_mask=True)
    #     t_fit = time.perf_counter() - t0
    #     fit_model_times.append(t_fit)
    #     if result is None:
    #         continue

    #     # --- Submit parallel tasks for segfit + eye plotting ---
    #     seg_future = executor.submit(seg_fit_renderer.render, frame, el_info, sclera_mask, result)
    #     plot_future = executor.submit(eye_move_vis.render, tp)
    #     # --- VisPy rendering (main thread only) ---
    #     t0 = time.perf_counter()
    #     center_panel = vispy_renderer.render(result)
    #     t_vispy = time.perf_counter() - t0
    #     vispy_renderer_times.append(t_vispy)

    #     # --- Collect parallel results ---
    #     t0 = time.perf_counter()
    #     left_panel = seg_future.result()
    #     t_seg = time.perf_counter() - t0
    #     seg_fit_renderer_times.append(t_seg)

    #     t0 = time.perf_counter()
    #     right_panel = plot_future.result()
    #     t_plot = time.perf_counter() - t0
    #     eye_move_vis_times.append(t_plot)

    #     combined = np.hstack([left_panel, center_panel, right_panel])
    #     writer.write(combined.astype(np.uint8))
    # writer.release()
    # executor.shutdown()

    # print(f"\nVideo saved to: {output_path}")

    # def compute_fps(times): return 1.0 / np.mean(times)

    # print("\nAverage FPS per component:")
    # print(f"  Model fitting     : {compute_fps(fit_model_times):.2f} fps")
    # print(f"  SegFit rendering  : {compute_fps(seg_fit_renderer_times):.2f} fps")
    # print(f"  VisPy rendering   : {compute_fps(vispy_renderer_times):.2f} fps")
    # print(f"  Signal plotting   : {compute_fps(eye_move_vis_times):.2f} fps")

