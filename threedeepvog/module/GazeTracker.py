import os
import torch
import threading
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pye3d.detector_3d import CameraModel, Detector3D, DetectorMode
from ..utils.cart2sph import cart2sph_batch_PL
from ..utils.read_and_save import save_json
from ..utils.transformation import PL2normDict_batch, circle2ellipse
'''
Currently, only implemented signle sphere eyeball model
The algorithm has optimized for vectorized computation in GPU
However, the accuracy is a bit lower than the original algorithm (LG), 
which is implemented in numpy, sequential computation  

conic projection have to use np.roots which is not supported by pytorch and cannot be used in GPU
this might be a bottleneck for the speed of the algorithm
'''

class GazeTracker(threading.Thread):
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
        self.mm2px = np.linalg.norm(np.array(self.args['resolution'])) / np.linalg.norm(np.array(self.args['sensor_size']))
   
        if self.args['focal_length_pxl'] is not None:
            self.fpx = self.args['focal_length_pxl'] 
        else:
            self.fpx = self.args['focal_length']*self.mm2px
        self.vertex = [0,0, -self.fpx]   # using for all unprojection and intersection
        self.img_shape = self.args['resolution']

        self.r_eye_default = 12.0 * self.mm2px  #distance betwween eyeball center to pupil center
        self.r_cornea_default = 7.8 * self.mm2px 
        self.r_iris_default = 6.0 * self.mm2px   # Iris ring radius

        self.pupil_dist = np.sqrt(self.r_eye_default**2 - self.r_iris_default**2)  #distance between eyeball center to pupil center
        self.re2dp = self.pupil_dist/self.r_eye_default
        self.max_eyeparams_opt_iters = self.args['max_eyeball_param_opt_iters']
        if self.eyeball_model == 'simple':
            self.r_pupil_default = 2.0 * self.mm2px    #2 * self.mm2px  (default: 1mm radius of pupil)
            self.default_eye_z = 50 * self.mm2px  #(default: 50mm distance from camera to eyeball)
        
        elif self.eyeball_model == 'LeGrand':
            self.r_pupil_default = 2.0 * self.mm2px    #2 * self.mm2px  (default: 1mm radius of pupil)
            self.default_eye_z = 35.0 * self.mm2px  #(default: 50mm distance from camera to eyeball)

        # List of parameters across a number (m) of observations
        self.gazes_unproj = [] # A list: ["gaze_positive"~np(m,3), "gaze_negative"~np(m,3)]
        self.c_pupil_unproj = [] # [ "pupil_3Dcentre_positive"~np(m,3), "pupil_3Dcentre_negative"~np(m,3) ]
        self.el_centres = None # reserved for numpy array (m,2) in numpy indexing frame,
        self.el_confs = None
        self.gazes_selected = None # reserved for (m,3) np.array in camera frame
        self.selected_pupil_positions = None  # reserved for (m,3) np.array in camera frame
        # Parameters of the eye model for consistent pupil estimate after initialisation
        self.c_eye2d = None # reserved for numpy array (2,1). Centre coordinate in numpy indexing frame.
        self.fit_residual = None # reserved for scalar. Residual of the fitting
        self.c_eye = None # reserved for (3,1) numpy array. 3D centre coordinate in camera frame
        self.r_eye = None # Scaler
        self.early_stop = False
        self.best_dice = 0.0

        self.fit_render = None

        if self.eyeball_model == 'PL':
            # self.detector_2d = Detector2D()
            self.camera = CameraModel(focal_length=self.fpx, resolution=self.args['resolution'])
            self.detector_3d = Detector3D(camera=self.camera, long_term_mode=DetectorMode.blocking)
            self.detector_3d.is_long_term_model_frozen = False
            self.frozen_count = 0
            self.frozen_min_frame = min(int(self.args['vid_nr_frames']/5), 1000)
            self.frozen_max_frame = int(self.args['vid_nr_frames'])
            self.frozen_count_thr = 80

        if self.args['mode'] == 'predict':
            eyeball_info = pd.read_json(self.args['eyeball_path'],orient='index').T
            if self.eyeball_model == 'simple' or self.eyeball_model == 'LeGrand':
                pass
            elif self.eyeball_model == 'PL':
                self.c_eye = eyeball_info.loc[0, ['eye_centre_x', 'eye_centre_y', 'eye_centre_z']].values
                self.r_eye = eyeball_info.loc[0, 'aver_eye_radius']
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
    
    
    def calc_model_iris_mask(self, result_3d):
        theta = np.linspace(0, 2 * np.pi, 100)
        xc, yc, zc = result_3d['circle_3d']['center']
        n = np.array(result_3d['circle_3d']['normal'])
        n /= np.linalg.norm(n)
        ref = np.array([0, 0, 1]) if np.abs(n[2]) < 0.99 else np.array([1, 0, 0])
        u = np.cross(n, ref); u /= np.linalg.norm(u)
        v = np.cross(n, u)
        r = 6.0
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        X_3d = xc + r * (cos_t * u[0] + sin_t * v[0])
        Y_3d = yc + r * (cos_t * u[1] + sin_t * v[1])
        Z_3d = zc + r * (cos_t * u[2] + sin_t * v[2])
        vid_w, vid_h = self.args['resolution']
        x_2d = self.fpx * (X_3d / Z_3d) + vid_w * 0.5
        y_2d = self.fpx * (Y_3d / Z_3d) + vid_h * 0.5
        # STEP 3: Rasterize to mask
        H, W = self.args['resolution'][1], self.args['resolution'][0]
        iris_proj_mask = np.zeros((H, W), dtype=np.uint8)
        pts = np.stack([x_2d, y_2d], axis=1).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(iris_proj_mask, [pts], color=1)
        return iris_proj_mask.astype(bool)
    
    
    def save_eyeball_model(self):
        if (self.c_eye is None) or (self.r_eye is None):
            print("3D eyeball model not found")
            raise Exception("3D eyeball model not found")
        else:
            save_dict = {
                        "eye_centre_x": self.c_eye[0],
                        "eye_centre_y": self.c_eye[1],
                        "eye_centre_z": self.c_eye[2],
                        "aver_eye_radius": self.r_eye,
                        "best_dice": self.best_dice,
                        "fit_residual": self.fit_residual
                        }
                        # Convert the data to Python-native types
        save_dict = {key: float(value) for key, value in save_dict.items()}
        self.args['eyeball_params'] = save_dict

        save_json(self.args['eyeball_path'], save_dict)
        print(f"Save eyeball model to {self.args['eyeball_path']}")
        

    def batch_fitting(self, frame_batch, mask=None): 
        time00 = time.time()
        self.batch_size = frame_batch['idxs'].shape[0]
        el_use = {key: frame_batch['ellipses'][f"{self.el_use}_{key}"] for key in ["center_x", "center_y", "w", "h", "radian", "confidence"]}
        el_pupil = {key: frame_batch['ellipses'][f"pupil_{key}"] for key in ["center_x", "center_y", "w", "h", "radian", "confidence"]}
        pred_iris_masks = frame_batch['iris_masks'].cpu().numpy()

        if self.eyeball_model == 'PL':
            grayscale_tensor = (frame_batch['imgs']*255).cpu()
            grayscale_array_batch_np = grayscale_tensor.byte().numpy()  # Convert to byte (uint8), required by pupilab function
            # df_el = pd.DataFrame(frame_batch['ellipses'])
            els = {key: value.cpu().numpy() for key, value in el_use.items() if isinstance(value, torch.Tensor)}

            for ix in range(self.batch_size):
                if (frame_batch['is_valid'][ix]) and not(frame_batch['blink'][ix]):
                    grayscale_array_np = grayscale_array_batch_np[ix]
                    timestamp = frame_batch['idxs'][ix].item() / self.args['vid_fps']
                    pupil_dict = self.create_pupil_dict(els, ix, timestamp)
                    result_3d = self.detector_3d.update_and_detect(pupil_dict, grayscale_array_np, apply_refraction_correction = True)
 
                    if not(self.detector_3d.is_long_term_model_frozen):
                        if (frame_batch['idxs'][ix]> self.frozen_min_frame):
                            # and (result_3d['model_confidence']==1):

                            # Compute Dice
                            model_iris_mask = self.calc_model_iris_mask(result_3d)
                            intersection = np.logical_and(model_iris_mask, pred_iris_masks[ix]).sum()
                            dice = (2.0 * intersection) / (model_iris_mask.sum() + pred_iris_masks[ix].sum() + 1e-8)
                            self.frozen_count += 1
                            self.best_dice = max(self.best_dice, dice)
                            if self.best_dice == dice:
                                self.c_eye = self.detector_3d.long_term_model.sphere_center   #uncorrected sphere center
                                self.r_eye = result_3d['sphere']['radius']
                                self.fit_residual = 0.0
                                if (self.frozen_count > self.frozen_count_thr) and (self.best_dice > self.dice_fitting_threshold):
                                    self.detector_3d.is_long_term_model_frozen = True
                                    self.threads['ques']['feedback'].put(True)
                                    self.early_stop = True

                            if (self.early_stop == False) and (frame_batch['idxs'][ix]==self.frozen_max_frame-1):
                                print(f"Eyeball fitting is not converged. Best dice: {self.best_dice}")
    
        elif self.eyeball_model == 'simple' or self.eyeball_model == 'LeGrand':
            pass
        self.frame_counter = self.frame_counter + self.batch_size 
        time01 = time.time()
        self.elapsed_time += (time01 - time00) 
        
        if (self.frame_counter == self.args['vid_nr_frames']-1) or self.early_stop:
            time00 = time.time()
            if (self.eyeball_model == 'simple') or (self.eyeball_model == 'LeGrand'):
                pass
            self.save_eyeball_model()
            time01 = time.time()
            self.elapsed_time += (time01 - time00) 

            
    def gaze_tracker(self, frame_batch):
        time00 = time.time()
        gaze_batch = []
        self.batch_size = frame_batch['idxs'].shape[0]
        el_use = {key: frame_batch['ellipses'][f"{self.el_use}_{key}"] for key in ["center_x", "center_y", "w", "h", "radian", "confidence"]}

        if self.eyeball_model == 'simple' or self.eyeball_model == 'LeGrand':
            pass

        elif self.eyeball_model == 'PL':
            # Convert to grayscale and scale on CPU
            frames_np = (frame_batch['imgs'].cpu().numpy()*255).astype(np.uint8)  # Convert to byte (uint8), required by pupilab function
            # df_el = pd.DataFrame(frame_batch['ellipses'])
            els = {key: value.cpu().numpy() for key, value in el_use.items() if isinstance(value, torch.Tensor)}
            results_3d = []
            for ix in range(self.batch_size):
                frame_np = frames_np[ix]
                tp = frame_batch['idxs'][ix].item() / self.args['vid_fps']
                result_3d = {"timestamp": tp}
                # ---- ellipse validity ----
                cx = els["center_x"][ix]
                cy = els["center_y"][ix]
                ww = els["w"][ix]
                hh = els["h"][ix]
                th = els["radian"][ix]
                conf = els["confidence"][ix]
                # ---- HARD FILTER ----
                if (not bool(frame_batch["is_valid"][ix])) or bool(frame_batch["blink"][ix]):
                    results_3d.append(result_3d)
                    self.frame_counter += 1
                    continue

                elif (conf <= 0) or np.isnan(cx) or np.isnan(cy) or np.isnan(ww) or np.isnan(hh) or np.isnan(th):
                    results_3d.append(result_3d)
                    self.frame_counter += 1
                    continue
                else:
                    pupil_dict = self.create_pupil_dict(els, ix, tp)
                    result_3d = self.detector_3d.update_and_detect(pupil_dict, frame_np, apply_refraction_correction = True)
                    
                    if self.frame_counter == -1:
                        # self.detector_3d.long_term_model.corrected_sphere_center = self.eye_centre
                        self.detector_3d.long_term_model.sphere_center = self.c_eye
                        self.detector_3d.is_long_term_model_frozen = True
                        self.detector_3d.long_term_model.corrected_sphere_center = \
                            self.detector_3d.long_term_model.refractionizer.correct_sphere_center(
                            np.asarray([[*self.c_eye]]))[0]
                        
                    norm_3d = self.normalize(result_3d["location"], (self.args['vid_w'], self.args['vid_h']), flip_y=True)
                    result_3d.update(self.create_summarize_dict(
                        norm_pos=norm_3d,
                        diameter=result_3d.get("diameter", 0.0),
                        confidence=result_3d.get("confidence", 0.0),
                        timestamp=tp,
                    ))
                    
                results_3d.append(result_3d)
                self.frame_counter += 1
            gaze_batch = PL2normDict_batch(results_3d)
            gaze_batch['hor'], gaze_batch['ver'] = cart2sph_batch_PL(gaze_batch['gaze'])
        time01 = time.time()
        self.elapsed_time += (time01 - time00) 
        # frame_batch['gaze_out'] = gaze_batch

        if self.use_queue:
            self.threads['ques']['gaze_out'].put(gaze_batch)


            if self.args['torsion_tracking_flag']:
                if self.args['torsion_geometric_correction_type']=='3D':

                    refpup_el, ok = circle2ellipse(torch.from_numpy(gaze_batch['c_pupil']).to(dtype=torch.float32, device=self.device),
                                                   torch.from_numpy(gaze_batch['gaze']).to(dtype=torch.float32, device=self.device),
                                                   torch.from_numpy(gaze_batch['r_pupil']).to(dtype=torch.float32, device=self.device),
                                                   self.fpx, (self.args['vid_h'], self.args['vid_w']))

                    iris_el, ok = circle2ellipse(torch.from_numpy(gaze_batch['c_pupil']).to(dtype=torch.float32, device=self.device),
                                                    torch.from_numpy(gaze_batch['gaze']).to(dtype=torch.float32, device=self.device),
                                                    torch.from_numpy(gaze_batch['r_iris']).to(dtype=torch.float32, device=self.device),
                                                    self.fpx, (self.args['vid_h'], self.args['vid_w']))
                    torsion_batch = {
                        'imgs': frame_batch['imgs'].to(dtype=torch.float32, device=self.device),
                        'gaze': torch.from_numpy(gaze_batch['gaze']).to(dtype=torch.float32, device=self.device),
                        'refpup_el': refpup_el,
                        'iris_el': iris_el,
                        'entpup_el': torch.from_numpy(gaze_batch['entpup_el']).to(dtype=torch.float32, device=self.device),
                        'c_eye': torch.from_numpy(self.c_eye).to(dtype=torch.float32, device=self.device),
                        'r_eye': float(self.r_eye),
                        'r_cornea': 7.8,
                        'r_iris': 6.0,
                        'is_valid': frame_batch['is_valid'],
                        'idxs': frame_batch['idxs'],
                        'blink': frame_batch['blink'],
                    }
                    self.threads['ques']['torsion_tracking'].put(torsion_batch)

            if self.args.get("fit_video_flag", False):
                B = frame_batch['imgs'].shape[0]
                self.threads['ques']['params_rendering'].put({
                    "frame_gray": frames_np,
                    "gaze_results": gaze_batch,
                    "torsion_results": np.zeros((B,1), dtype=np.float32),
                })
        else:
            return gaze_batch  
        

    def run(self):
        while True:
            frame_batch = self.threads['ques']['gaze_tracking'].get()

            if frame_batch is None:
                self.threads['ques']['gaze_out'].put(None)

                if self.args.get('torsion_tracking_flag'):
                    if self.args.get('torsion_geometric_correction_type') == '3D':
                        self.threads['ques']['torsion_tracking'].put(None)
                else:
                    if self.args.get("fit_video_flag", False):
                        self.threads['ques']['fitted_frame_out'].put(None)
                break

            if not self.early_stop:
                if self.args['mode'] == 'fit':
                    self.batch_fitting(frame_batch)
                elif self.args['mode'] == 'predict':
                    self.gaze_tracker(frame_batch)
                else:
                    raise ValueError("Unknown mode for GazeTracker_LG")
                