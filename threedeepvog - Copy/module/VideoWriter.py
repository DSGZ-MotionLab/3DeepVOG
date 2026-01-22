import numpy as np
import threading
import skvideo.io as skv
import cv2
import subprocess



'''
Currently, only implemented signle sphere eyeball model
The algorithm has optimized for vectorized computation in GPU
However, the accuracy is a bit lower than the original algorithm (LG), 
which is implemented in numpy, sequential computation  

conic projection have to use np.roots which is not supported by pytorch and cannot be used in GPU
this might be a bottleneck for the speed of the algorithm
'''

import os
# import sys    # sys.path.append("D:/git/DeepVOG3DTorch/DeepVOG/deepvog3D")
import torch
import logging
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from monai.transforms import Resize
import skvideo.io as skv
import kornia.enhance as kornia_enhance

from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
import subprocess    #Run a command at python script  (e.g. ls, dir, etc.)
import threading
import queue
# import plotly.offline as pyo
# from itertools import product
from ..tool.unprojection import convert_ell_to_general_batch, unprojectGazePositions_batch, reproject, reverse_reproject
from ..tool.intersection import intersect ,intersect_batch, fit_ransac_batch, fit_ransac_batch_v2, fit_ransac_batch_v3, fit_ransac_batch_v4, fit_ransac_batch_v5, fit_ransac_batch_v6, line_sphere_intersect_batch
from ..tool.cart2sph import cart2sph_batch
from ..utils.read_and_save import save_json
from ..utils.visualization import gen_sphere_mesh, fit_legrand_model
# from utils.visualization import fit_legrand_model, project_3d_to_2d


class FitVideoWriter(threading.Thread):
    def __init__(self, threads, args, daemon=False, use_queue=True):
        super().__init__(daemon=daemon)
        self.dtype = torch.float64   #for accurate calculation
        # torch.set_default_dtype(self.dtype)
    
        self.name = 'Gaze-Tracking'
        self.threads = threads
        self.args = args
        self.use_queue = use_queue  # Flag to determine if using queues
        self.device = args['device']
        self.eyeball_model = args['eyeball_model']   # 'simple' or 'LeGrand' or 'PL'
        self.elapsed_time = 0
        self.frame_counter = -1
        self.el_use = "pupil"
        self.confidence_fitting_threshold = self.args['threshold_confidence_pupil'] if self.el_use == "pupil" else self.args['threshold_confidence_iris']
        self.dice_fitting_threshold = 0.95
        self.circularity_max = 0.985   # exclude circle like ellipse -> error-prone fitted ellipse
        self.circularity_min = 0.5   # exclude extreme ellipse -> high eye tilted angle
        self.mm2px_scaling = np.linalg.norm(np.array(self.args['resolution'])) / np.linalg.norm(np.array(self.args['sensor_size']))
   
        if self.args['focal_length_pxl'] is not None:
            self.focal_length = self.args['focal_length_pxl'] 
        else:
            self.focal_length = self.args['focal_length']*self.mm2px_scaling
        self.vertex = [0,0, -self.focal_length]   # using for all unprojection and intersection
        self.image_shape = self.args['resolution']

        self.defult_eyeball_radius = 12.0 * self.mm2px_scaling  #distance betwween eyeball center to pupil center
        self.defult_cornia_radius = 7.8 * self.mm2px_scaling 
        self.defult_limbus_radius = 6.0 * self.mm2px_scaling   # Iris ring radius

        self.pupil_dist = np.sqrt(self.defult_eyeball_radius**2 - self.defult_limbus_radius**2)  #distance between eyeball center to pupil center
        self.re2dp = self.pupil_dist/self.defult_eyeball_radius
        self.max_eyeparams_opt_iters = self.args['max_eyeball_param_opt_iters']
        if self.eyeball_model == 'simple':
            self.defult_pupil_radius = 2.0 * self.mm2px_scaling    #2 * self.mm2px_scaling  (default: 1mm radius of pupil)
            self.default_eye_z = 50 * self.mm2px_scaling  #(default: 50mm distance from camera to eyeball)
        
        elif self.eyeball_model == 'LeGrand':
            self.defult_pupil_radius = 2.0 * self.mm2px_scaling    #2 * self.mm2px_scaling  (default: 1mm radius of pupil)
            self.default_eye_z = 35.0 * self.mm2px_scaling  #(default: 50mm distance from camera to eyeball)

        # List of parameters across a number (m) of observations
        self.unproj_gaze_vectors = [] # A list: ["gaze_positive"~np(m,3), "gaze_negative"~np(m,3)]
        self.unproj_pupil_centres = [] # [ "pupil_3Dcentre_positive"~np(m,3), "pupil_3Dcentre_negative"~np(m,3) ]
        self.ellipse_centres = None # reserved for numpy array (m,2) in numpy indexing frame,
        self.ellipse_confidences = None
        self.selected_gazes = None # reserved for (m,3) np.array in camera frame
        self.selected_pupil_positions = None  # reserved for (m,3) np.array in camera frame
        # Parameters of the eye model for consistent pupil estimate after initialisation
        self.proj_eye_centre = None # reserved for numpy array (2,1). Centre coordinate in numpy indexing frame.
        self.fit_residual = None # reserved for scalar. Residual of the fitting
        self.eye_centre = None # reserved for (3,1) numpy array. 3D centre coordinate in camera frame
        self.best_dice = 0.0
        self.aver_eye_radius = None # Scaler
        self.early_stop = False

        self.eyeball_params = {
            'eyeball_radius': 12, 'corneaball_radius': 7.8, 'limbus_radius': 6.0,
            'dp': np.sqrt(12**2 - 6.0**2), 'num_grids': 25,
            'line_thickness': max(1, int(np.round((np.linalg.norm(self.args['resolution'])/500)))),
            'eyeball_center': self.eye_centre,
        }

        self.camera_params = {
            'fcl_mm': self.args['focal_length'],   #in mm
            'mm2px_scaling': self.mm2px_scaling,
            'resolution': self.args['resolution'],
        }

        self.color_map = self.args.get("color_map", {   
                        "eyeball_mesh": (255, 150, 50), "corneaball_mesh": (255, 150, 255),
                        "limbus_circle_points": (255, 0, 255), "pupil_circle_points": (255, 255, 0),
                        "entpup_el": (144, 238, 144),  
                        "eyeball_center": (0, 0, 255), "pupil_center": (0, 255, 127), 
                        "gaze_vector": (0, 255, 127),
                    })
        
        if self.eyeball_model == 'PL':
            # self.detector_2d = Detector2D()
            self.camera = CameraModel(focal_length=self.focal_length, resolution=self.args['resolution'])
            self.detector_3d = Detector3D(camera=self.camera, long_term_mode=DetectorMode.blocking)
            self.detector_3d.is_long_term_model_frozen = False
            self.frozen_count = 0
            self.frozen_min_frame = min(int(self.args['vid_nr_frames']/5), 1000)
            self.frozen_max_frame = int(self.args['vid_nr_frames'])
            self.frozen_count_thr = 80

        if not(self.args['mode'] == 'fit'):
            eyeball_info = pd.read_json(self.args['eyeball_path'],orient='index').T
            if self.eyeball_model == 'simple' or self.eyeball_model == 'LeGrand':
                self.eye_centre = torch.tensor(eyeball_info.loc[0, ['eye_centre_x', 'eye_centre_y', 'eye_centre_z']].values.reshape(3,1), dtype=self.dtype, device=self.device)
                self.aver_eye_radius = eyeball_info.loc[0, 'aver_eye_radius']
                self.proj_eye_centre = reproject(self.eye_centre, self.focal_length, batch_mode= False)
            elif self.eyeball_model == 'PL':
                self.eye_centre = eyeball_info.loc[0, ['eye_centre_x', 'eye_centre_y', 'eye_centre_z']].values
                self.aver_eye_radius = eyeball_info.loc[0, 'aver_eye_radius']
                self.best_dice = eyeball_info.loc[0, 'best_dice']
    
    #from pupil_src/shared_modules/methods.py
    def normalize(self, pos, size, flip_y=False) -> float:
        x, y = pos[0]/float(size[0]), pos[1]/float(size[1])
        return x, 1 - y if flip_y else x, y

    def create_pupil_dict(self, els, ix, timestamp):
        return {
            "ellipse": {
                "center": [els['center_x'][ix], els['center_y'][ix]],
                "axes": [els['w'][ix]*2, els['h'][ix]*2],
                "angle": np.rad2deg(els['radian'][ix]),   #require degree input
            },
            "confidence": els['confidence'][ix],
            "timestamp": timestamp,
        }

    #from pupil_src/shared_modules/pupil_detector_plugins/detector_base_plugin.py
    def create_summarize_dict(self, norm_pos, diameter, confidence, timestamp) -> dict:
        return {
            "norm_pos": norm_pos,
            "diameter": diameter,
            "confidence": confidence,
            "timestamp": timestamp,
        }
        
    def norm_vec_batch(self, vec_batch):
        return vec_batch/torch.linalg.norm(vec_batch, axis=1).reshape(-1,1)
    
    

    def _project_3d_to_2d(self, X, Y, Z):
        w, h = self.camera_params['resolution']
        fcl, scale = self.camera_params['fcl_mm'], self.camera_params['mm2px_scaling']
        return (fcl * (X / Z) * scale + w * 0.5, fcl * (Y / Z) * scale + h * 0.5)
    

    def render_model_fitting(self, result, image):
        projections = {k: self._project_3d_to_2d(*v) for k, v in result.items()
                        if k in ["eyeball_mesh", "corneaball_mesh", "limbus_circle_points", "pupil_circle_points"]}
        eye2d = self._project_3d_to_2d(*result["eyeball_center"])
        pupil2d = self._project_3d_to_2d(*result["pupil_center"])
        n = self.eyeball_params['num_grids']
        t = self.eyeball_params['line_thickness']

        for mesh, color in zip(["eyeball_mesh", "corneaball_mesh"],
                                [self.color_map["eyeball_mesh"], self.color_map["corneaball_mesh"]]):
            x, y = projections[mesh]
            for i in range(n):
                for pts in [np.column_stack((x[i], y[i])), np.column_stack((x[:, i], y[:, i]))]:
                    pts = pts[~np.isnan(pts).any(axis=1)]
                    if len(pts) > 1:
                        cv2.polylines(image, [pts.astype(np.int32)], False, color, t)
        for key in ["limbus_circle_points", "pupil_circle_points"]:
            x, y = projections[key]
            pts = np.column_stack((x, y)).astype(np.int32)
            cv2.polylines(image, [pts], True, self.color_map[key], t + 1)
        cv2.circle(image, tuple(map(int, eye2d)), t + 2, self.color_map["eyeball_center"], -1)
        cv2.circle(image, tuple(map(int, pupil2d)), t + 2, self.color_map["pupil_center"], -1)

    
        theta, cx, cy, a, b = result["entpup_el"] 
        center = (int(round(cx)), int(round(cy)))
        axes = (int(round(a)), int(round(b)))   # expects semi-axes
        angle = float(np.degrees(theta))
        cv2.ellipse(image, center, axes, angle, 0, 360, self.color_map["entpup_el"], t + 1, lineType=cv2.LINE_AA)

        if "gaze_vec" in result:
            end = (int(pupil2d[0] + 50 * result["gaze_vec"][0]), int(pupil2d[1] + 50 * result["gaze_vec"][1]))
            cv2.line(image, tuple(map(int, pupil2d)), end, self.color_map["gaze_vector"], t * 2, lineType=cv2.LINE_AA)
        return image


            
    def gaze_tracker(self, frame_batch):
        time00 = time.time()
        gaze_batch = []
        self.batch_size = frame_batch['idxs'].shape[0]
        el_use = {key: frame_batch['ellipses'][f"{self.el_use}_{key}"] for key in ["center_x", "center_y", "w", "h", "radian", "confidence"]}

        if self.eyeball_model == 'simple' or self.eyeball_model == 'LeGrand':
            mask = frame_batch['is_valid']
            # self.use = "pupil" or "iris"
            self.frame_counter = self.frame_counter + self.batch_size 
            
        elif self.eyeball_model == 'PL':
            # Convert to grayscale and scale on CPU
            grayscale_tensor = (frame_batch['imgs']*255).cpu()
            grayscale_array_batch_np = grayscale_tensor.byte().numpy()  # Convert to byte (uint8), required by pupilab function
            # df_el = pd.DataFrame(frame_batch['ellipses'])
            els = {key: value.cpu().numpy() for key, value in el_use.items() if isinstance(value, torch.Tensor)}
            fitted_frame_all = []
            for ix in range(self.batch_size):
                if (els['confidence'][ix] == 0) or (np.isnan(els['center_x'][ix])) or\
                    (np.isnan(els['center_y'][ix])) or (np.isnan(els['w'][ix])) or\
                    (np.isnan(els['h'][ix])) or (np.isnan(els['radian'][ix])):
    
                    result_3d = dict()
                    result_3d['timestamp'] = frame_batch['idxs'][ix].item() / self.args['vid_fps']
                    result_3d['sphere'] = {'center': (0.0, 0.0, 0.0), 'radius': 0.0}
                    result_3d['projected_sphere'] = {'center': (0.0, 0.0), 'axes': (0.0, 0.0), 'angle': 0.0}
                    result_3d['circle_3d'] = {'center': (0.0, 0.0, 0.0), 'normal': (0.0, 0.0, 0.0), 'radius': 0.0}
                    result_3d['diameter_3d'] = 0
                    result_3d['ellipse'] = {'center': (0.0, 0.0), 'axes': (0.0, 0.0), 'angle': 0.0}
                    result_3d['location'] = (0.0, 0.0)
                    result_3d['diameter'] = 0
                    result_3d['confidence'] = 0
                    result_3d['model_confidence'] = 0
                    result_3d['theta'] = 0.0
                    result_3d['phi'] = 0.0
                    result_3d['norm_pos'] = (0.0, 0.0)

                    if self.args['write_fit_video']:
                        frame = (frame_batch['imgs'][ix].cpu().numpy()*255).astype(np.uint8)
                        fitted_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # -> (H,W,3) BGR
                else:
                    timestamp = frame_batch['idxs'][ix].item() / self.args['vid_fps']
                    pupil_dict = self.create_pupil_dict(els, ix, timestamp)
                    grayscale_array_np = grayscale_array_batch_np[ix]
                    result_3d = self.detector_3d.update_and_detect(pupil_dict, grayscale_array_np, apply_refraction_correction = True)
                    
                    if self.frame_counter == -1:
                        # self.detector_3d.long_term_model.corrected_sphere_center = self.eye_centre
                        self.detector_3d.long_term_model.sphere_center = self.eye_centre
                        self.detector_3d.is_long_term_model_frozen = True
                        self.detector_3d.long_term_model.corrected_sphere_center = \
                            self.detector_3d.long_term_model.refractionizer.correct_sphere_center(
                            np.asarray([[*self.eye_centre]]))[0]
                        
                    norm_3d = self.normalize(result_3d["location"], (self.args['vid_w'], self.args['vid_h']), flip_y=True)
                    temp_3d = self.create_summarize_dict(
                        norm_pos=norm_3d,
                        diameter=result_3d["diameter"],
                        confidence= result_3d["confidence"],     # definition is black box, can't identify
                        timestamp= timestamp,
                    )
                    result_3d.update(temp_3d)

                    # --- Output setup ---
                    if self.args['write_fit_video']:
                        r_e, r_c, r_s = self.eyeball_params['eyeball_radius'], self.eyeball_params['corneaball_radius'], self.eyeball_params['limbus_radius']
                        d_p, d_lim_cornea = np.sqrt(r_e**2 - r_s**2), np.sqrt(r_c**2 - r_s**2)
                        limbus_theta_eye, limbus_theta_cornea = np.arccos(d_p / r_e), np.arccos(d_lim_cornea / r_c)
                        n_grids = self.eyeball_params['num_grids']

                        df_gaze_out = pd.DataFrame([result_3d])
                        processed_torsion = 0.0
                        large_meshes = gen_sphere_mesh(r_e, n_grids, insert_theta=limbus_theta_eye)
                        small_meshes = gen_sphere_mesh(r_c, n_grids, insert_theta=limbus_theta_cornea)
                        result = fit_legrand_model(self.eyeball_params, df_gaze_out.iloc[0], processed_torsion,
                                                large_meshes, small_meshes, n_grids, use_mask=True)
                        frame = (frame_batch['imgs'][ix].cpu().numpy()*255).astype(np.uint8)
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # -> (H,W,3) BGR
                        fitted_frame = self.render_model_fitting(result, frame.copy())

                if self.args['write_fit_video']:
                    fitted_frame_all.append(fitted_frame)
                gaze_batch.append(result_3d)
                self.frame_counter += 1

        time01 = time.time()
        self.elapsed_time += (time01 - time00) 
        # frame_batch['gaze_out'] = gaze_batch

        if self.use_queue:
            self.threads['ques']['gaze_out'].put(gaze_batch)
            if self.args['write_fit_video']:
                fitted_frame_all = np.array(fitted_frame_all)
                self.threads['ques']['fitted_frame_out'].put(fitted_frame_all)
        else:
            return gaze_batch
        

    def run(self):
        while True:
            # read data from source queue
            frame_batch = self.threads['ques']['gaze_tracking'].get()
            if frame_batch is None: # received poison pill, pass on and quit!
                self.threads['ques']['gaze_out'].put(None)    
                if self.args['do_torsion_tracking'] and self.args['torsion_geometric_correction_type']=='3D':
                    self.threads['ques']['torsion_tracking'].put(None)

                if self.args['viz_gaze']:
                    pass
                #     self.threads['ques']['torsion_visualization'].put(None)
                # print('%s: received poison pill! Closing!'%(self.name))
                break

            if not(self.early_stop):
                # if self.params['is_fit']:
                if self.args['mode'] == 'fit':
                    self.batch_fitting(frame_batch)
                else:
                    self.gaze_tracker(frame_batch)





# def round_up_to_odd_int(f):
#     return int(np.ceil(f) // 2 * 2 + 1)

# def visualization_merger(params, method='ffmpeg'):
#     '''
#     - read queues for viz_seg, viz_pupil_params, viz_ellipse, viz_gaze, viz_torsion
#     - vstack the images
#     - send frame_batch to VideoWriter.writeFrame
#     '''
#     vid_files = []
#     if params['viz_segmentation']:
#         vid_files.append(params['viz_filename_mp4_ellipses'])
#     if params['viz_gaze']:
#         pass
#     if params['viz_torsion']:
#         vid_files.append(params['viz_filename_mp4_torsion'])
    
#     if len(vid_files) > 1:
#         if method == 'ffmpeg':
#             cmd_ffmpeg = [
#                 'ffmpeg'
#             ]
#             for filepath in vid_files:
#                 cmd_ffmpeg.extend(['-i', filepath])
#             cmd_ffmpeg.extend([
#                 '-filter_complex', f'vstack=inputs={len(vid_files)}',
#                 params['viz_filename_mp4']
#             ])
#             print('visualization_merger ffmpeg cmd:')
#             print(' '.join(cmd_ffmpeg))
#             ret_val = subprocess.run(cmd_ffmpeg)
#         else:
#             ret_val = -1
#     else: 
#         ret_val = -1
#     return ret_val

# class VideoWriter(threading.Thread):
#     def __init__(self, threads, args, use_ffmpeg=True, filename_override=None, src_que='video_writer_ellipses', daemon=False, use_queue=True):
#         super().__init__(daemon=daemon)
#         self.name = 'Thread-' + src_que
#         self.threads = threads
#         self.params = args
#         self.use_queue = args.get('is_parallel', use_queue)  #priority to args['is_parallel'] if it exists
#         self.elapsed_time = 0
#         self.fps = args['vid_fps']
#         self.viz_frame_interval = args['viz_frame_interval']
#         self.src_que = src_que
#         self.use_ffmpeg = use_ffmpeg

#         if args['viz_frame_interval'] > 1:
#             self.fps = self.fps / args['viz_frame_interval']

#         self.viz_filename_mp4 = args['viz_filename_mp4']
#         if filename_override is not None:
#             self.viz_filename_mp4 = filename_override

#         if self.use_ffmpeg:
#             self.inputdict = {'-r': str(self.fps)}
#             self.outputdict = {
#                 '-vcodec': 'libx264',
#                 '-pix_fmt': 'yuv420p',
#                 '-r': str(self.fps),
#                 '-crf': '15'
#             }
#             self.vwriter = skv.FFmpegWriter(
#                 self.viz_filename_mp4,
#                 inputdict=self.inputdict,
#                 outputdict=self.outputdict
#             )
#         else:
#             fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#             self.vwriter = cv2.VideoWriter(self.viz_filename_mp4, fourcc, self.fps, (args['vid_w'], args['vid_h']))

#     def run(self):
#         while True:
#             frame_batch = self.threads['ques'][self.src_que].get()
#             if frame_batch is None: 
#                 print('Videowriter "%s": received poison pill! Closing vwriter.' % self.name)
#                 self.release()
#                 break
#             else:
#                 if self.viz_frame_interval == 1:
#                     self.write_frame_batch(frame_batch)
#                 else:
#                     self.write_single_frame(frame_batch)
    
#     def write_single_frame(self, viz_frame):
#         if self.use_ffmpeg:
#             self.vwriter.writeFrame(viz_frame['img'])
#         else:
#             self.vwriter.write(viz_frame['img'])

#         if viz_frame['idx'] % 100 == 0:
#             pass
#             # print(f'Wrote frame {viz_frame["idx"]} to vid ({self.viz_filename_mp4})')

#     def write_frame_batch(self, viz_frames):
#         for idx, viz_frame in enumerate(viz_frames):
#             if self.use_ffmpeg:
#                 self.vwriter.writeFrame(viz_frame['img'])
#             else:
#                 self.vwriter.write(viz_frame['img'])

#             if viz_frame['idx'] % 100 == 0:
#                 pass
#                 # print(f'Wrote frame {viz_frame["idx"]} to vid ({self.viz_filename_mp4})')

#     def release(self):
#         print('Releasing video writer...')
#         if self.use_ffmpeg:
#             self.vwriter.close()
#         else:
#             self.vwriter.release()