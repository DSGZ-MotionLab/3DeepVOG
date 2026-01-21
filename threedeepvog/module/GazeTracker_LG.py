
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
# from pupil_detectors import Detector2D
# from pye3d.refraction import Refractionizer

# from .cpp.refraction_correction import apply_correction_pipeline
# refractionizer = Refractionizer()
# print(refractionizer.correct_sphere_center([[0.0, 0.0, 35.0]]))
# input_features = np.asarray(
#     [[*self.sphere_center, *pupil_circle.normal, pupil_circle.radius]]
# )
# refraction_corrected_params = self.refractionizer.correct_pupil_circle(
#     input_features
# )[0]

import subprocess    #Run a command at python script  (e.g. ls, dir, etc.)
import threading
import queue


# import plotly.offline as pyo
# from itertools import product
from ..tool.unprojection import convert_ell_to_general_batch, unprojectGazePositions_batch, reproject, reverse_reproject
from ..tool.intersection import intersect ,intersect_batch, fit_ransac_batch, fit_ransac_batch_v2, fit_ransac_batch_v3, fit_ransac_batch_v4, fit_ransac_batch_v5, fit_ransac_batch_v6, line_sphere_intersect_batch
from ..tool.cart2sph import cart2sph_batch
from ..utils.read_and_save import save_json
from ..utils.transformation import PL2normDict_batch
from ..utils.visualization import gen_sphere_mesh, fit_legrand_model
# from utils.visualization import fit_legrand_model, project_3d_to_2d

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
            self.focal_length = self.args['focal_length_pxl'] 
        else:
            self.focal_length = self.args['focal_length']*self.mm2px
        self.vertex = [0,0, -self.focal_length]   # using for all unprojection and intersection
        self.image_shape = self.args['resolution']

        self.defult_eyeball_radius = 12.0 * self.mm2px  #distance betwween eyeball center to pupil center
        self.defult_cornia_radius = 7.8 * self.mm2px 
        self.defult_limbus_radius = 6.0 * self.mm2px   # Iris ring radius

        self.pupil_dist = np.sqrt(self.defult_eyeball_radius**2 - self.defult_limbus_radius**2)  #distance between eyeball center to pupil center
        self.re2dp = self.pupil_dist/self.defult_eyeball_radius
        self.max_eyeparams_opt_iters = self.args['max_eyeball_param_opt_iters']
        if self.eyeball_model == 'simple':
            self.defult_pupil_radius = 2.0 * self.mm2px    #2 * self.mm2px  (default: 1mm radius of pupil)
            self.default_eye_z = 50 * self.mm2px  #(default: 50mm distance from camera to eyeball)
        
        elif self.eyeball_model == 'LeGrand':
            self.defult_pupil_radius = 2.0 * self.mm2px    #2 * self.mm2px  (default: 1mm radius of pupil)
            self.default_eye_z = 35.0 * self.mm2px  #(default: 50mm distance from camera to eyeball)

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

        self.fit_render = None

        if self.eyeball_model == 'PL':
            # self.detector_2d = Detector2D()
            self.camera = CameraModel(focal_length=self.focal_length, resolution=self.args['resolution'])
            self.detector_3d = Detector3D(camera=self.camera, long_term_mode=DetectorMode.blocking)
            self.detector_3d.is_long_term_model_frozen = False
            self.frozen_count = 0
            self.frozen_min_frame = min(int(self.args['vid_nr_frames']/5), 1000)
            self.frozen_max_frame = int(self.args['vid_nr_frames'])
            self.frozen_count_thr = 80

        if self.args['mode'] == 'predict':
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
    
    
    def unproject_batch_observation(self, el_use, valid_ix): 
        # Convert centre coordinates from numpy indexing frame to camera frames
        # circularity = ellipses['pupil_w']/ellipses['pupil_h']
        el_use['center_x'][~valid_ix] = torch.nan
        el_use['center_y'][~valid_ix] = torch.nan
        el_use['w'][~valid_ix] = torch.nan
        el_use['h'][~valid_ix] = torch.nan
        el_use['radian'][~valid_ix] = torch.nan

        #TODO: should be float 64 in ellipse fitting class
        centre = torch.vstack([el_use['center_x'], el_use['center_y']]).to(self.dtype)
        w, h, radian = el_use['w'].to(self.dtype), el_use['h'].to(self.dtype), el_use['radian'].to(self.dtype)
        centre_cam = centre.clone()
        centre_cam[0] = centre_cam[0] - self.image_shape[0]/2
        centre_cam[1] = centre_cam[1] - self.image_shape[1]/2
        # Convert ellipse parameters to the coefficients of the general form of ellipse equation
        A,B,C,D,E,F = convert_ell_to_general_batch(centre_cam[0],centre_cam[1], w, h, radian)
        ell_co = (A,B,C,D,E,F)

        # Unproject the ellipse to obtain 2 ambiguous gaze vectors with numpy shape (3,1),
        # and pupil_centre with numpy shape (3,1)
        unproj_gaze_pos, unproj_gaze_neg , unproj_pupil_centre_pos, unproj_pupil_centre_neg = unprojectGazePositions_batch(self.vertex, ell_co, self.defult_pupil_radius, valid_ix)
        unproj_gaze_pos = self.norm_vec_batch(unproj_gaze_pos)
        unproj_gaze_neg = self.norm_vec_batch(unproj_gaze_neg)
        
        return unproj_gaze_pos, unproj_gaze_neg, unproj_pupil_centre_pos, unproj_pupil_centre_neg, centre.T
    

    #based on information theory, the more data you have, the more accurate the model will be
    def fit_projected_eye_centre(self, ransac = False, max_iters = 1000, min_distance = 2000):
    # You will need to determine when to fit outside of the class
        if (self.unproj_gaze_vectors is None) or (self.ellipse_centres is None):
            raise TypeError('No unprojected gaze lines or ellipse centres were added (not yet initalized). Use add_to_fitting() function to add them first.')

        # self.unprojected_3D_pupil_positions is more noisy
        a = torch.vstack((self.ellipse_centres,self.ellipse_centres))  
        # a = torch.vstack([self.unprojected_3D_pupil_positions[:,0,0:2], self.unprojected_3D_pupil_positions[:,0,0:2]])
        n = torch.vstack((self.unproj_gaze_vectors[:,0,0:2], self.unproj_gaze_vectors[:,1,0:2]))
        if ransac == True:
            abnormality_check = self.ellipse_centres.shape[0]
            samples_to_fit = int(a.shape[0]/8)    #quater of survived samples for each minibatch fitting
            weights = self.ellipse_confidences.to(torch.float64)
            w = torch.sigmoid(5*(weights - torch.median(weights)))
            w = torch.hstack([w,w])
            # self.proj_eye_centre, self.fit_residual = fit_ransac_batch_v5(a,n, batch_size = max_iters, sample_size = samples_to_fit, prior_weight = 0.1, weights = w)
            self.proj_eye_centre = fit_ransac_batch_v6(a,n, batch_size = max_iters, sample_size = samples_to_fit) 
            self.fit_residual = 0

        if ransac == False: # or self.eyeball_model == 'LeGrand':
            self.proj_eye_centre = intersect_batch(a.unsqueeze(0), n.unsqueeze(0), device=self.device)
        if (self.proj_eye_centre is None):
            raise TypeError('You did not fit a eyeball model.')
        if abnormality_check != self.ellipse_centres.shape[0]:
            print("GPU overloading detected. You should reduce the number of samples to fit.")
        return self.proj_eye_centre

    def fit_LeGrands_eye_centre(self, ransac = False, c_tilde = None):
        # matrix = sum_aux_3d[:3, :3]
            # You will need to determine when to fit outside of the class
        if (self.selected_gazes is None) or (self.selected_pupil_positions is None):
            raise TypeError('No unprojected gaze lines or ellipse centres were added (not yet initalized). Use add_to_fitting() function to add them first.')
        a = self.selected_pupil_positions - torch.tensor(self.pupil_dist)*self.selected_gazes   #/self.mm2px    #dn
        n = self.norm_vec_batch(self.selected_pupil_positions)  #q  
        # Normalisation of the 2D projection of gaze vectors is done inside intersect()
        if ransac == True:
            max_iters = 10000
            samples_to_fit = int(a.shape[0]/10) 
            self.eye_centre = fit_ransac_batch_v6(a, n, batch_size = max_iters, sample_size = samples_to_fit)
            if c_tilde is not None:
                eye_centre_z = torch.mm(c_tilde.T, self.eye_centre/torch.mm(c_tilde.T, c_tilde)).item()
                self.eye_centre = eye_centre_z * c_tilde
        else:
            self.eye_centre = intersect_batch(a.unsqueeze(0), n.unsqueeze(0), device=self.device)
        if (self.eye_centre is None):
            raise TypeError('You did not fit a eyeball model.')
        # intersect_batch(a[5000:8000:200,:].unsqueeze(0), n[5000:8000:200,:].unsqueeze(0), device=self.device)/self.mm2px
        # self._corrected_sphere_center = self.refractionizer.correct_sphere_center(
        #     np.asarray([[*self.sphere_center]])
        # )[0]
        return self.eye_centre

    def estimate_eye_sphere(self):
        # This function is called once after fit_projected_eye_centre()
        # self.initial_eye_z is required (in pixel unit)
        # self.initial_eye_z shall be the z-distance between the point and camera vertex (in camera frame)
        if (self.proj_eye_centre is None):
            # pdb.set_trace()
            raise TypeError('Projected_eye_centre must be initialized first')
    
        # Unprojecting the 2D projected eye centre to 3D.
        # Converting the projected_eye_centre from numpy indexing frame to camera frame
        proj_eye_centre_camera_frame = self.proj_eye_centre.clone()
        proj_eye_centre_camera_frame[0] = proj_eye_centre_camera_frame[0] - self.image_shape[0]/2
        proj_eye_centre_camera_frame[1] = proj_eye_centre_camera_frame[1] - self.image_shape[1]/2
        
        # Reconstructed selected gaze vectors and pupil positions by rejecting those pointing away from projected eyecentre
        gazes = [self.unproj_gaze_vectors[:,0,:], self.unproj_gaze_vectors[:,1,:]]
        positions = [self.unproj_pupil_centres[:,0,:], self.unproj_pupil_centres[:,1,:]]
        selected_gazes, selected_positions = self.disambiguate_dierkes_lines_batch(gazes, positions, proj_eye_centre_camera_frame)   
        self.selected_gazes, self.selected_pupil_positions = selected_gazes, selected_positions

        if self.eyeball_model == 'simple':
        # Unprojection: Nearest intersection of two lines. 
        # a = [eye_centre, pupil_3Dcentre], n =[gaze_vector, pupil_3D_centre]
            proj_eye_centre_camera_frame_scaled = \
                reverse_reproject(proj_eye_centre_camera_frame, self.default_eye_z, self.focal_length)  #camera coord
            eye_centre_camera_frame = torch.cat(
                [proj_eye_centre_camera_frame_scaled.flatten(), 
                torch.tensor([self.default_eye_z], device=self.device, dtype=self.dtype)]
            ).view(3, 1)

            eye_centre_camera_frame_expanded = eye_centre_camera_frame.T.unsqueeze(0).repeat(selected_positions.shape[0], 1, 1)  # Shape [668, 1, 3]
            selected_positions_expanded = selected_positions.unsqueeze(1)  # Shape [668, 1, 3]
            a_3Dfitting_batch = torch.cat((eye_centre_camera_frame_expanded, selected_positions_expanded), dim=1)  # Shape [668, 2, 3]
            n_3Dfitting_batch = torch.cat((selected_gazes.unsqueeze(1), selected_positions_expanded), dim=1)  # Shape [668, 2, 3]
            # self.aver_eye_radius, radius_counter = fit_eyeball_radius_batch(a_3Dfitting_batch, n_3Dfitting_batch, eye_centre_camera_frame)
            intersected_pupil_3D_centre_batch = intersect_batch(a_3Dfitting_batch, n_3Dfitting_batch) 
            radius_batch = torch.linalg.norm(intersected_pupil_3D_centre_batch - eye_centre_camera_frame.unsqueeze(0), dim=1)  # Shape [668, 3]
            radius_counter = radius_batch.shape[0]
            self.aver_eye_radius = torch.mean(radius_batch.squeeze()).item()    #*self.re2dp
            self.eye_centre = eye_centre_camera_frame.cpu().numpy()

            eye_z = ((self.default_eye_z*self.pupil_dist)/self.aver_eye_radius)
            self.aver_eye_radius = self.pupil_dist
            unproj_xy_coord = \
                reverse_reproject(proj_eye_centre_camera_frame, eye_z, self.focal_length)  #camera coord
            
            self.eye_centre = torch.cat(
                [unproj_xy_coord.flatten(), 
                torch.tensor([eye_z], device=self.device, dtype=self.dtype)]
            ).view(3, 1).cpu().numpy()


        elif self.eyeball_model == 'LeGrand':   #use algortihm from Dirkes, but not staible
            # worst_distance = np.linalg.norm(self.params['resolution'])   #3 * num_frames_fit
            # harmonious_eye_centre = eye_centre_camera_frame/self.default_eye_z
            # self.eye_centre = self.fit_LeGrands_eye_centre(ransac=True, c_tilde = harmonious_eye_centre) 
            self.eye_centre = self.fit_LeGrands_eye_centre(ransac=True) 
            self.aver_eye_radius = self.pupil_dist
            radius_counter = 1
        return self.aver_eye_radius, radius_counter
    
    
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
                        "best_dice": self.best_dice,
                        "fit_residual": self.fit_residual
                        }
                        # Convert the data to Python-native types
        save_dict = {key: float(value) for key, value in save_dict.items()}
        self.args['eyeball_params'] = save_dict

        save_json(self.args['eyeball_path'], save_dict)
        print(f"Save eyeball model to {self.args['eyeball_path']}")

    def disambiguate_dierkes_lines_batch(self, gazes, positions, projected_centre):
    # gazes is a list ~ [gaze_vector_pos~(3,1), gaze_vector_neg~(3,1)]  (n^+ and n^-)
    # positions is a list ~ [pupil_position_pos~(3,1), pupil_position_neg~(3,1)] (p^+ and p^-)
    # projected_centre ~ numpy array~(2,1) (c)
    # Disambiguaion of the gaze vectors and pupil positions  -- EQ(8) Lech Swirski et al. 2013
        selected_gaze = gazes[0]
        selected_position = positions[0]
        # projected_gaze = reproject(selected_position + selected_gaze, self.focal_length, batch_mode= True) - projected_centre.T   #why this definition? n^~ = (p^~ + n^~) - c^~
        projected_gaze = reproject(selected_gaze, self.focal_length, batch_mode= True)
        projected_position = reproject(selected_position, self.focal_length, batch_mode= True)  #p^~
        dot_products = torch.sum(projected_gaze * (projected_position - projected_centre.T), dim=1)
        flip_ix = dot_products < 0    #n^~*(p^~ - c^~) > 0 -> select p+, n+, else select p-, n-
        selected_gaze[flip_ix] = gazes[1][flip_ix,:]
        selected_position[flip_ix] = positions[1][flip_ix,:]
        return selected_gaze, selected_position
        
    def calc_3Dpupil_info_batch(self):
        # This function must be called after using unproject_single_observation() to update surrent observation
        if (self.eye_centre is None) or (self.aver_eye_radius is None):
            raise TypeError("Call estimate_eye_sphere() to initialize eye_centre and eye_radius first.")
        else:
            selected_gaze_batch, selected_position_batch = \
                  self.disambiguate_dierkes_lines_batch([self.current_pos_gazes, self.current_neg_gazes], 
                                                        [self.current_pupil_pos_centres, self.current_pupil_neg_centres],
                                                        self.proj_eye_centre)
            o = torch.zeros((3,1), dtype=self.dtype, device=self.device)
            consistence = torch.ones(self.batch_size, dtype=torch.bool)

            d1, d2, invalid_mask = line_sphere_intersect_batch(self.eye_centre, 
                                                               torch.tensor(self.aver_eye_radius), 
                                                               o, 
                                                               self.norm_vec_batch(selected_position_batch))
            
            new_pos_min = o.T + torch.min(d1,d2)*self.norm_vec_batch(selected_position_batch)
            # new_pos_max = o.T + torch.max(d1,d2)*self.norm_vec_batch(selected_position_batch)
            new_radius_min = (self.defult_pupil_radius/selected_position_batch[:,-1])*new_pos_min[:,-1]
            # new_radius_max = (self.defult_pupil_radius/selected_position_batch[:,-1])*new_pos_max[:,-1]
            new_gaze_min = self.norm_vec_batch(new_pos_min - self.eye_centre.T)
            # new_gaze_max = self.norm_vec_batch(new_pos_max - self.eye_centre.T)

            # print("Cannot find line-sphere interception. Old pupil parameters are used.")
            new_pos_min[invalid_mask] = selected_position_batch[invalid_mask]
            new_gaze_min[invalid_mask] = selected_gaze_batch[invalid_mask]
            new_radius_min[invalid_mask] = self.defult_pupil_radius
            consistence[invalid_mask] = False
            return new_pos_min, new_gaze_min, new_radius_min, consistence
            # return [new_pos_min, new_pos_max], [new_gaze_min, new_gaze_max], [new_radius_min, new_radius_max], consistence


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
                                self.eye_centre = self.detector_3d.long_term_model.sphere_center   #uncorrected sphere center
                                self.aver_eye_radius = result_3d['sphere']['radius']
                                self.fit_residual = 0.0
                                if (self.frozen_count > self.frozen_count_thr) and (self.best_dice > self.dice_fitting_threshold):
                                    self.detector_3d.is_long_term_model_frozen = True
                                    self.threads['ques']['feedback'].put(True)
                                    self.early_stop = True

                                # #debugging
                                # plt.imshow(grayscale_array_batch_np[ix], alpha=0.5)
                                # plt.imshow(model_iris_mask.astype(float), alpha=0.5, cmap='gray')
                                # plt.imshow(pred_iris_masks[ix].astype(float), alpha=0.5, cmap='gray')
                                # plt.savefig("debug_output.png")  # Save to file instead of showing

                            if (self.early_stop == False) and (frame_batch['idxs'][ix]==self.frozen_max_frame-1):
                                print(f"Eyeball fitting is not converged. Best dice: {self.best_dice}")
    
        elif self.eyeball_model == 'simple' or self.eyeball_model == 'LeGrand':
            circularity = el_use['w']/el_use['h']
            mask = frame_batch['is_valid'] & ~frame_batch['blink'] &\
                (el_use['confidence'] > self.confidence_fitting_threshold) & \
                (circularity < self.circularity_max) & (circularity > self.circularity_min) \

            if not(all(~mask)):
                #This is all camera centered)
                self.current_pos_gazes, self.current_neg_gazes,\
                self.current_pupil_pos_centres, self.current_pupil_neg_centres,\
                self.ellipse_batch_centres = self.unproject_batch_observation(el_use, mask)

                if (len(self.unproj_gaze_vectors)==0) or (len(self.unproj_pupil_centres) ==0) or (self.ellipse_centres is None):
                    self.ellipse_confidences = el_use['confidence'][mask]
                    self.unproj_gaze_vectors = torch.stack([self.current_pos_gazes[mask], self.current_neg_gazes[mask]], dim=1)
                    self.unproj_pupil_centres = torch.stack([self.current_pupil_pos_centres[mask], self.current_pupil_neg_centres[mask]], dim=1)
                    self.ellipse_centres = self.ellipse_batch_centres[mask]
                else:
                    self.ellipse_confidences = torch.hstack([self.ellipse_confidences, el_use['confidence'][mask]])
                    self.unproj_gaze_vectors = torch.vstack([self.unproj_gaze_vectors, torch.stack([self.current_pos_gazes[mask], self.current_neg_gazes[mask]], dim=1)])
                    self.unproj_pupil_centres = torch.vstack([self.unproj_pupil_centres, torch.stack([self.current_pupil_pos_centres[mask], self.current_pupil_neg_centres[mask]], dim=1)])
                    self.ellipse_centres = torch.vstack([self.ellipse_centres, self.ellipse_batch_centres[mask]])

        self.frame_counter = self.frame_counter + self.batch_size 
        time01 = time.time()
        self.elapsed_time += (time01 - time00) 
        
        if (self.frame_counter == self.args['vid_nr_frames']-1) or self.early_stop:
            time00 = time.time()
            if (self.eyeball_model == 'simple') or (self.eyeball_model == 'LeGrand'):
                worst_dist = np.linalg.norm(self.args['resolution'])
                self.fit_projected_eye_centre(ransac=True, max_iters=self.args['max_eyeball_param_opt_iters'], min_distance= worst_dist)
                self.estimate_eye_sphere()
            self.save_eyeball_model()
            time01 = time.time()
            self.elapsed_time += (time01 - time00) 

            
    def gaze_tracker(self, frame_batch):
        time00 = time.time()
        gaze_batch = []
        self.batch_size = frame_batch['idxs'].shape[0]
        el_use = {key: frame_batch['ellipses'][f"{self.el_use}_{key}"] for key in ["center_x", "center_y", "w", "h", "radian", "confidence"]}

        if self.eyeball_model == 'simple' or self.eyeball_model == 'LeGrand':
            mask = frame_batch['is_valid']
            # self.use = "pupil" or "iris"
            if not(all(~mask)):
                self.current_pos_gazes, self.current_neg_gazes,\
                self.current_pupil_pos_centres,self.current_pupil_neg_centres,\
                _ = self.unproject_batch_observation(el_use, mask)
                p_batch, n_batch, pupil_radius_batch, consistence_batch = self.calc_3Dpupil_info_batch()
                theta_batch, phi_batch = cart2sph_batch(n_batch.T)  
                p_batch = p_batch.cpu().numpy()
                n_batch = n_batch.cpu().numpy()
                gaze_batch = {
                    "hor": theta_batch.cpu().numpy(),
                    "ver": phi_batch.cpu().numpy(),
                    "c_pupil": p_batch,   # Bx3
                    "gaze": n_batch,   # Bx3
                    "confidence": el_use['confidence'].cpu().numpy(),
                    "consistence": consistence_batch.cpu().numpy(),
                    "r_pupil": pupil_radius_batch.cpu().numpy()
                }
            else: 
                gaze_batch = {
                    "hor": np.zeros(self.batch_size),
                    "ver": np.zeros(self.batch_size),
                    "c_pupil": np.zeros(self.batch_size,3),
                    "gaze": np.zeros(self.batch_size,3),
                    "confidence": np.zeros(self.batch_size),
                    "consistence": np.zeros(self.batch_size),
                    "r_pupil": np.zeros(self.batch_size)
                }
            self.frame_counter = self.frame_counter + self.batch_size 
            
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
                # if self.args['fit_video_flag']:
                #     fitted_frame = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)  # -> (H,W,3) BGR

                if (els['confidence'][ix] == 0) or (np.isnan(els['center_x'][ix])) or\
                    (np.isnan(els['center_y'][ix])) or (np.isnan(els['w'][ix])) or\
                    (np.isnan(els['h'][ix])) or (np.isnan(els['radian'][ix])):
                    pass
                else:
                    pupil_dict = self.create_pupil_dict(els, ix, tp)
                    result_3d = self.detector_3d.update_and_detect(pupil_dict, frame_np, apply_refraction_correction = True)
                    
                    if self.frame_counter == -1:
                        # self.detector_3d.long_term_model.corrected_sphere_center = self.eye_centre
                        self.detector_3d.long_term_model.sphere_center = self.eye_centre
                        self.detector_3d.is_long_term_model_frozen = True
                        self.detector_3d.long_term_model.corrected_sphere_center = \
                            self.detector_3d.long_term_model.refractionizer.correct_sphere_center(
                            np.asarray([[*self.eye_centre]]))[0]
                        
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

            self.threads['ques']['gaze_out'].put(gaze_batch)
        time01 = time.time()
        self.elapsed_time += (time01 - time00) 
        # frame_batch['gaze_out'] = gaze_batch

        if self.use_queue:
            self.threads['ques']['gaze_out'].put(gaze_batch)
            # if self.args['fit_video_flag']:
            #         fitted_frame_all = np.array(fitted_frame_all)
                    # self.threads['ques']['fitted_frame_out'].put(fitted_frame_all)

            if self.args['torsion_tracking_flag']:
                if self.args['torsion_geometric_correction_type']=='3D':
                    torsion_batch = {
                        'useful_maps': None,
                        'is_valid': self.is_valid,
                        # 'ellipses': el_dicts, gaze_batch
                        'idxs': frame_batch['idxs'],
                        # 'blink': blink
                    }
                    self.threads['ques']['torsion_tracking'].put(torsion_batch)
            # else:
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
                