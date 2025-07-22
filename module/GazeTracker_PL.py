
#%% script_05_dv3d_threaded_classes.py
# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
import threading
import json
# # from itertools import product

'''
Currently, only implemented PL two-sphere eyeball model
The algorithm has optimized for CPU sequential computation which was implemented in numpy  
'''
def save_json(path, save_dict):
    json_str = json.dumps(save_dict, indent=4)
    with open(path, "w") as fh:
        fh.write(json_str)
        

class GazeTracker(threading.Thread):
    def __init__(self, threads, args, daemon=False, use_queue=True):
        super().__init__(daemon=daemon)
        self.dtype = torch.float64   #for accurate calculation
        # torch.set_default_dtype(self.dtype)
    
        self.name = 'Gaze-Tracking'
        self.threads = threads
        self.args = args
        self.use_queue = args.get('is_parallel', use_queue)  #priority to args['is_parallel'] if it exists
        self.device = args['device']
        self.eyeball_model = args['eyeball_model']   # 'simple' or 'LeGrand' or 'PL'
        self.elapsed_time = 0
        self.frame_counter = -1
        self.el_use = "pupil"   #"pupil" or "iris"
        self.confidence_fitting_threshold = self.args['threshold_confidence_pupil'] if self.el_use == "pupil" else self.args['threshold_confidence_iris']
        self.dice_fitting_threshold = 0.95
        self.circularity_max = 0.985   # exclude circle like ellipse -> error-prone fitted ellipse
        self.circularity_min = 0.5   # exclude extreme ellipse -> high eye tilted angle
        self.mm2px_scaling = self.args['mm2px_scaling']   # scaling factor from mm to pixel
   
        if self.args['focal_length_pxl'] is not None:
            self.focal_length = self.args['focal_length_pxl'] 
        else:
            self.focal_length = self.args['focal_length']*self.mm2px_scaling
        self.vertex = [0,0, -self.focal_length]   # using for all unprojection and intersection
        self.image_shape = self.args['resolution']

        self.defult_eyeball_radius = 12.0 * self.mm2px_scaling  #distance betwween eyeball center to pupil center
        self.defult_cornia_radius = 7.8 * self.mm2px_scaling 
        self.defult_limbus_radius = 6.0 * self.mm2px_scaling   # Iris ring radius
        self.defult_pupil_radius = 2.0 * self.mm2px_scaling    #2 * self.mm2px_scaling  (default: 1mm radius of pupil)

        self.pupil_dist = np.sqrt(self.defult_eyeball_radius**2 - self.defult_limbus_radius**2)  #distance between eyeball center to pupil center
        self.re2dp = self.pupil_dist/self.defult_eyeball_radius
        self.max_eyeparams_opt_iters = self.args['max_eyeball_param_opt_iters']

        # List of parameters across a number (m) of observations
        self.norms_pn_3d = [] # A list: ["gaze_positive"~np(m,3), "gaze_negative"~np(m,3)]
        self.centres_pn_3d = [] # [ "pupil_3Dcentre_positive"~np(m,3), "pupil_3Dcentre_negative"~np(m,3) ]
        self.ellipse_centres = None # reserved for numpy array (m,2) in numpy indexing frame,
        self.ellipse_confidences = None
        self.selected_gazes = None # reserved for (m,3) np.array in camera frame
        self.selected_pupil_positions = None  # reserved for (m,3) np.array in camera frame
        # Parameters of the eye model for consistent pupil estimate after initialisation
        self.proj_eye_centre = None # reserved for numpy array (2,1). Centre coordinate in numpy indexing frame.
        self.fit_residual = None # reserved for scalar. Residual of the fitting
        self.eye_centre = None # reserved for (3,1) numpy array. 3D centre coordinate in camera frame
        self.aver_eye_radius = None # Scaler
        self.early_stop = False

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
            self.eye_centre = eyeball_info.loc[0, ['eye_centre_x', 'eye_centre_y', 'eye_centre_z']].values
            self.aver_eye_radius = eyeball_info.loc[0, 'aver_eye_radius']
    
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
        x_2d = self.focal_length * (X_3d / Z_3d) + vid_w * 0.5
        y_2d = self.focal_length * (Y_3d / Z_3d) + vid_h * 0.5
        # STEP 3: Rasterize to mask
        H, W = self.args['resolution'][1], self.args['resolution'][0]
        iris_proj_mask = np.zeros((H, W), dtype=np.uint8)
        pts = np.stack([x_2d, y_2d], axis=1).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(iris_proj_mask, [pts], color=1)
        return iris_proj_mask.astype(bool)
    

    def batch_fitting(self, frame_batch, mask=None, threshold = 0.5): 
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
                        if (frame_batch['idxs'][ix]> self.frozen_min_frame) and \
                            (result_3d['model_confidence']==1):
                            model_iris_mask = self.calc_model_iris_mask(result_3d)
                            # STEP 4: Compute Dice
                            intersection = np.logical_and(model_iris_mask, pred_iris_masks[ix]).sum()
                            dice = (2.0 * intersection) / (model_iris_mask.sum() + pred_iris_masks[ix].sum() + 1e-8)

                            self.frozen_count += 1
                            if (self.frozen_count > self.frozen_count_thr) and (dice > self.dice_fitting_threshold):
                                self.detector_3d.is_long_term_model_frozen = True
                                self.eye_centre = self.detector_3d.long_term_model.sphere_center   #uncorrected sphere center
                                self.aver_eye_radius = result_3d['sphere']['radius']
                                self.fit_residual = 0
                                self.threads['ques']['feedback'].put(True)
                                self.early_stop = True

                                # #debugging
                                # plt.imshow(grayscale_array_batch_np[ix], alpha=0.5)
                                # plt.imshow(model_iris_mask.astype(float), alpha=0.5, cmap='gray')
                                # plt.imshow(pred_iris_masks[ix].astype(float), alpha=0.5, cmap='gray')
                                # plt.savefig("debug_output.png")  # Save to file instead of showing

                            if (frame_batch['idxs'][ix]>self.frozen_max_frame):
                                pass
                        elif (self.early_stop == False) and (frame_batch['idxs'][ix]>self.frozen_max_frame):
                            self.eye_centre = self.detector_3d.long_term_model.sphere_center   #uncorrected sphere center
                            self.aver_eye_radius = result_3d['sphere']['radius']
                            print(f"Eyeball fitting is not converged. Use the last eyeball model. fitting dice: {dice}")
    
        elif self.eyeball_model == 'simple' or self.eyeball_model == 'LeGrand':
            circularity = el_use['w']/el_use['h']
            mask = frame_batch['is_valid'] & ~frame_batch['blink'] &\
                (el_use['confidence'] > self.confidence_fitting_threshold) & \
                (circularity < self.circularity_max) & (circularity > self.circularity_min) \

            if not(all(~mask)):
                #This is all camera centered)
                self.unproj_gaze_pos, self.unproj_gaze_neg,\
                self.unproj_centre_pos, self.unproj_centre_neg,\
                centres_ellipses = self.unproject_batch_observation(el_use, mask)

                if (len(self.norms_pn_3d)==0) or (len(self.centres_pn_3d) ==0) or (self.ellipse_centres is None):
                    self.ellipse_confidences = el_use['confidence'][mask]
                    self.norms_pn_3d = torch.stack([self.unproj_gaze_pos[mask], self.unproj_gaze_neg[mask]], dim=1)
                    self.centres_pn_3d = torch.stack([self.unproj_centre_pos[mask], self.unproj_centre_neg[mask]], dim=1)
                    self.ellipse_centres = centres_ellipses[mask]
                else:
                    self.ellipse_confidences = torch.hstack([self.ellipse_confidences, el_use['confidence'][mask]])
                    self.norms_pn_3d = torch.vstack([self.norms_pn_3d, torch.stack([self.unproj_gaze_pos[mask], self.unproj_gaze_neg[mask]], dim=1)])
                    self.centres_pn_3d = torch.vstack([self.centres_pn_3d, torch.stack([self.unproj_centre_pos[mask], self.unproj_centre_neg[mask]], dim=1)])
                    self.ellipse_centres = torch.vstack([self.ellipse_centres, centres_ellipses[mask]])

        self.frame_counter = self.frame_counter + self.batch_size 
        time01 = time.time()
        self.elapsed_time += (time01 - time00) 
        
        if (self.frame_counter == self.args['vid_nr_frames']-1) or self.early_stop:
            time00 = time.time()
            # if (self.eyeball_model == 'simple') or (self.eyeball_model == 'LeGrand'):
            #     worst_dist = np.linalg.norm(self.args['resolution'])
            #     self.fit_projected_eye_centre(ransac=True, max_iters=self.args['max_eyeball_param_opt_iters'], min_distance= worst_dist)
            #     self.estimate_eye_sphere()
            self.save_eyeball_model()
            time01 = time.time()
            self.elapsed_time += (time01 - time00) 


    def save_eyeball_model(self):
        if (self.eye_centre is None) or (self.aver_eye_radius is None):
            print("3D eyeball model not found")
            raise Exception("3D eyeball model not found")
        else:
            save_dict = {
                        "eye_centre_x": self.eye_centre[0],
                        "eye_centre_y": self.eye_centre[1],
                        "eye_centre_z": self.eye_centre[2],
                        "aver_eye_radius": self.aver_eye_radius,
                        "fit_residual": self.fit_residual
                        }
                        # Convert the data to Python-native types
        save_dict = {key: float(value) for key, value in save_dict.items()}
        # save_path = os.path.join(self.params['vid_root'], 
        #                          f"{self.params['vid_base_name']}_{self.eyeball_model}_eyeball_model.json")
        
        # self.params['eyeball_params'] = save_dict
        # self.params['eyeball_path'] = self.params.get('eyeball_path', save_path)
        save_json(self.args['eyeball_path'], save_dict)
        print(f"Save eyeball model to {self.args['eyeball_path']}")

        

    def run(self):
        while True:
            # read data from source queue
            frame_batch = self.threads['ques']['gaze_tracking'].get()
            if frame_batch is None: # received poison pill, pass on and quit!
                self.threads['ques']['gaze_out'].put(None)    
                if self.args['do_torsion_tracking'] and self.args['torsion_geometric_correction_type']=='3D':
                    self.threads['ques']['torsion_tracking'].put(None)

                # if self.args['viz_gaze']:
                #     pass
                #     self.threads['ques']['torsion_visualization'].put(None)
                # print('%s: received poison pill! Closing!'%(self.name))
                break

            if not(self.early_stop):
                if self.args['mode'] == 'fit':
                    self.batch_fitting(frame_batch)
                else:
                    self.gaze_tracker(frame_batch)
    

    def gaze_tracker(self, frame_batch):
        time00 = time.time()
        gaze_batch = []
        self.batch_size = frame_batch['idxs'].shape[0]
        el_use = {key: frame_batch['ellipses'][f"{self.el_use}_{key}"] for key in ["center_x", "center_y", "w", "h", "radian", "confidence"]}

        if self.eyeball_model == 'PL':
            # # # Convert to grayscale and scale on CPU
            # pred_iris_masks = frame_batch['iris_masks'].cpu().numpy()
            # pred_pupil_masks = frame_batch['pupil_masks'].cpu().numpy()

            grayscale_tensor = (frame_batch['imgs']*255).cpu()
            grayscale_array_batch_np = grayscale_tensor.byte().numpy()  # Convert to byte (uint8), required by pupilab function
            # df_el = pd.DataFrame(frame_batch['ellipses'])
            els = {key: value.cpu().numpy() for key, value in el_use.items() if isinstance(value, torch.Tensor)}
            
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
                    
                    # # #debugging  
                    # model_iris_mask = self.calc_model_iris_mask(result_3d)
                    # # STEP 4: Compute Dice
                    # intersection = np.logical_and(model_iris_mask, pred_iris_masks[ix]).sum()
                    # dice = (2.0 * intersection) / (model_iris_mask.sum() + pred_iris_masks[ix].sum() + 1e-8)
                    # plt.figure(figsize=(10, 10))
                    # plt.imshow(grayscale_array_np, cmap='gray')
                    # plt.imshow(model_iris_mask.astype(float), cmap='gray', alpha=0.3)
                    # plt.imshow(pred_iris_masks[ix].astype(float), cmap='gray', alpha=0.3)
                    # plt.savefig(f"model_iris.png")

                gaze_batch.append(result_3d)
                self.frame_counter += 1

        time01 = time.time()
        self.elapsed_time += (time01 - time00) 
        # frame_batch['gaze_out'] = gaze_batch

        if self.use_queue:
            self.threads['ques']['gaze_out'].put(gaze_batch)
            # if self.args['viz_gaze']:
            #     pass
                # frame_batch_gaze_viz = {'idxs': frame_batch['idxs'],
                #                     'gaze': gaze_batch}
                # self.threads['ques']['gaze_visualization'].put(frame_batch_gaze_viz)
        else:
            return gaze_batch