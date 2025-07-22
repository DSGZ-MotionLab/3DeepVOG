
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import os, torch, time, cv2, threading
# import sys    # sys.path.append("D:/git/DeepVOG3DTorch/DeepVOG/deepvog3D")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import torch.nn.functional as F
# plt.ion()  # Enable interactive mode
from monai.transforms import Resize
import skvideo.io as skv
import kornia.enhance as kornia_enhance
import plotly.offline as pyo
# from itertools import product
# import skvideo.io as skv
from skimage.color import label2rgb
from skimage import measure
from astropy.convolution import convolve as nan_convolve

# from itertools import product

# import skvideo.io as skv
from skimage.color import label2rgb
from skimage import measure
from astropy.convolution import convolve as nan_convolve
# from scipy.spatial import ConvexHull


class TorsionTracker(threading.Thread):
    def __init__(self, threads, args, daemon=False, use_queue=True):
        super().__init__(daemon=daemon)
        self.name = 'Thread-TorsionTracker'
        self.threads = threads
        self.params = args
        self.use_queue = args.get('is_parallel', use_queue)  #priority to args['is_parallel'] if it exists
        self.frame_counter = -1
        # general params
        # device is currently hard-coded to cpu, as gpu performs ~4x slower
        self.device = args['device']
        self.vid_w = args['vid_w']
        self.vid_h = args['vid_h']
        # torsion-specific params
        self.update_interval = int(args['torsion_force_update_template_interval_sec']*args['vid_fps'])
        self.angular_pxl2deg = args['torsion_angular_pxl2deg']
        self.n_subimgs = args['torsion_n_subimgs']
        self.pad_tempW_pxl = int(np.ceil(args['torsion_max_deg']/self.angular_pxl2deg))  #number of pxl padding to temp polar iris map
        self.subimg_w_pxl  = int(np.ceil(args['torsion_subimg_angular_deg']/self.angular_pxl2deg)) #20
        self.subimg_h_pxl   = args['torsion_subimg_h_pxl']
        self.img_h_pxl = args['torsion_radial_pxl']    #pxl height size of polar iris pattern map
        args['torsion_angular_pxl'] = int(360/self.angular_pxl2deg)
        self.img_w_pxl  = int(360/self.angular_pxl2deg)   #pxl width size of polar iris pattern map
        self.cover_threshold = args['torsion_coverage_rate_threshold']
        self.TM_algorithm = args['torsion_TM_algorithm']
        args['torsion_polarmap_h_pxl'] = self.img_h_pxl
        args['torsion_polarmap_w_pxl']  = self.img_w_pxl
        self.unpdate_flag = False
        self.elapsed_time = 0
        self.polar_elapsed_time = 0
        self.TM_elapsed_time = 0

        self.mm2px_scaling = np.linalg.norm(np.array(self.params['resolution'])) / np.linalg.norm(np.array(self.params['sensor_size']))
        self.canonical_eyeball_radius = 12.0*self.mm2px_scaling

        # some initial values
        self.template_available = False
        self.template_iris_img = torch.zeros((self.img_h_pxl,self.img_w_pxl),dtype=torch.float32,device=self.device)
        self.template_frame_idx = -1
        self.template_quality = {'polar_coverage_percent': -1.0,
                                 'polar_xgradient_sum': -1.0,
                                 'polar_xgradient_average': -1.0}

        # storage for torsion angle values
        self.template_torsion_offset = np.nan
        self.torsion_angles = []        
        self.current_iris_imgs = []

        if self.params['torsion_geometric_correction_type']=='3D':
            eyeball_info = pd.read_json(self.params['eyeball_path'],orient='index').T
            self.eye_centre = eyeball_info.loc[0, ['eye_centre_x', 'eye_centre_y', 'eye_centre_z']].values.reshape(3,1)
            self.eye_radius = eyeball_info.loc[0, 'aver_eye_radius']

        # self.compiled_ncc = torch.compile(self.ncc_batch)
    def run(self):
        while True:
            # read data from source queue
            frame_batch = self.threads['ques']['torsion_tracking'].get()
            if self.params['torsion_geometric_correction_type']=='3D':
                # eyeball_info = self.threads['ques']['eyeball_info'].get()
                pass
            # if frame_batch['imgs'].shape[0]<32:
            if frame_batch is None: # received poison pill, pass on and quit!
                self.threads['ques']['torsion_out'].put(None)  
                # self.threads['ques']['torsion_distribution'].put(None)      
                # if self.params['viz_torsion']:
                #     self.threads['ques']['torsion_visualization'].put(None)
                # print('%s: received poison pill! Closing!'%(self.name))
                break
            else:
                self.torsion_tracker(frame_batch)
                
    
    def normalizeRobust_batch(self, img_batch, percentiles=[5, 95], bg_threshold=0.05):
        # Calculate the percentiles & mask for valid pixels
        if img_batch.dim() == 2:
            img_batch = img_batch.unsqueeze(0)
        q = torch.tensor(percentiles, dtype=torch.float32, device=self.device) / 100.0
        valid_pxl_mask = img_batch > bg_threshold
        # Replace invalid pixels with a very large value for quantile calculation
        valid_pxls = torch.where(valid_pxl_mask, img_batch, torch.tensor(torch.nan, device=self.device))
        th_low, th_high = torch.nanquantile(valid_pxls.view(img_batch.shape[0], -1), q, dim=-1, keepdim=True).squeeze()
        # Handle cases where valid pixels are not present
        th_low[th_low == torch.nan] = 0
        th_high[th_high == torch.nan] = 1
        # Rescale intensities & Clip the values
        img_batch -= th_low.reshape(-1, 1, 1)
        img_batch /= (th_high.reshape(-1, 1, 1) - th_low.reshape(-1, 1, 1))
        img_batch = torch.clip(img_batch, 0.0, 1.0)

        # Determine success for each image
        success_batch = ((th_high - th_low) > 0).squeeze()
        # if success_batch[i] is False, then img_batch[i] is zero map
        img_batch = torch.where(success_batch.view(-1, 1, 1), img_batch, torch.zeros_like(img_batch))
        return img_batch, success_batch
    
    @staticmethod
    def ellipse2affine_batch(el, device):
        theta, w, h, loc_x, loc_y = (el['pupil_radian'], el['pupil_w'], el['pupil_h'], el['pupil_center_x'], el['pupil_center_y'])
        skew_x = 2 * w / (w + h)
        skew_y = 2 * h / (w + h)
        # Create the rotation matrices
        Arot_pre_batch = torch.stack([
            torch.stack([torch.cos(-theta), -torch.sin(-theta), torch.zeros_like(theta)], dim=1),
            torch.stack([torch.sin(-theta), torch.cos(-theta), torch.zeros_like(theta)], dim=1),
            torch.tensor([0, 0, 1], dtype=torch.float32, device=device).repeat(theta.size(0), 1)
        ], dim=1)

        Arot_batch = torch.stack([
            torch.stack([torch.cos(theta), -torch.sin(theta), torch.zeros_like(theta)], dim=1),
            torch.stack([torch.sin(theta), torch.cos(theta), torch.zeros_like(theta)], dim=1),
            torch.tensor([0, 0, 1], dtype=torch.float32, device=device).repeat(theta.size(0), 1)
        ], dim=1)

        # Create skew matrices
        Askew_batch = torch.stack([
            torch.stack([skew_x, torch.zeros_like(skew_x), torch.zeros_like(skew_x)], dim=1),
            torch.stack([torch.zeros_like(skew_y), skew_y, torch.zeros_like(skew_y)], dim=1),
            torch.tensor([0, 0, 1], dtype=torch.float32, device=device).repeat(theta.size(0), 1)
        ], dim=1)

        # Create scale matrices (assuming scale parameters are identity matrices)
        Ascale_batch = torch.eye(3, dtype=torch.float32, device=device).repeat(theta.size(0), 1, 1)

        # Create translation matrices
        Atrans_batch = torch.stack([
            torch.stack([torch.ones_like(loc_x), torch.zeros_like(loc_x), loc_x], dim=1),
            torch.stack([torch.zeros_like(loc_y), torch.ones_like(loc_y), loc_y], dim=1),
            torch.tensor([0, 0, 1], dtype=torch.float32, device=device).repeat(theta.size(0), 1)
        ], dim=1)

        # Perform the matrix multiplications for all 32 matrices
        A_batch = torch.bmm(torch.bmm(torch.bmm(torch.bmm(Atrans_batch, Arot_batch), Askew_batch), Ascale_batch), Arot_pre_batch)
        return A_batch
    
    @staticmethod
    def polish_el2affine_batch(el, B, H, device):
        """Returns affine transform matrices [B * H, 3, 3] for pupil ellipses."""
        # B, device = self.batch_size, self.device
        theta = el['pupil_radian'].view(B, 1)
        epsilon = 0.1
        stp = torch.linspace(0, 1, H, device=device)
        # theta = el['pupil_radian']   #should I also expand this?
        sx_pup = (2 * el['pupil_w'] / (el['pupil_w'] + el['pupil_h'])).unsqueeze(1)
        sy_pup = (2 * el['pupil_h'] / (el['pupil_w'] + el['pupil_h'])).unsqueeze(1)
        sx_iris = (2 * el['iris_w'] / (el['iris_w'] + el['iris_h'])).unsqueeze(1)
        sy_iris = (2 * el['iris_h'] / (el['iris_w'] + el['iris_h'])).unsqueeze(1)
        cx_pup, cy_pup = el['pupil_center_x'].unsqueeze(1), el['pupil_center_y'].unsqueeze(1)
        cx_iris, cy_iris = el['iris_center_x'].unsqueeze(1), el['iris_center_y'].unsqueeze(1)
        cx_delta, cy_delta = (cx_iris - cx_pup)*(1 - epsilon), (cy_iris - cy_pup)*(1 - epsilon)
        sx_delta, sy_delta = (sx_iris - sx_pup)*(1 - epsilon), (sy_iris - sy_pup)*(1 - epsilon)
        # Generate meshgrid using broadcasting
        x_shift = (cx_pup + cx_delta * stp)  #torch.Size([B, self.img_h_pxl])
        y_shift = (cy_pup + cy_delta * stp)  
        x_skew = (sx_pup + sx_delta * stp)
        y_skew = (sy_pup + sy_delta * stp)

        # Expand for shape [B, H]
        for var in [x_shift, y_shift, x_skew, y_skew]:
            var.expand(B, H)
        cos_t = torch.cos(theta).expand(B, H)
        sin_t = torch.sin(theta).expand(B, H)
        z = torch.zeros(B, H, device=device)
        o = torch.ones(B, H, device=device)
        br = torch.tensor([0, 0, 1], device=device).repeat(B * H, 1).view(B, H, 3)
        Arot = torch.stack([torch.stack([cos_t, -sin_t, z], -1), torch.stack([sin_t,  cos_t, z], -1), br], -2)
        Arot_pre = torch.stack([torch.stack([ cos_t, sin_t, z], -1), torch.stack([-sin_t, cos_t, z], -1), br], -2)
        Askew = torch.stack([torch.stack([x_skew, z, z], -1), torch.stack([z, y_skew, z], -1), br], -2)
        Atrans = torch.stack([torch.stack([o, z, x_shift], -1),torch.stack([z, o, y_shift], -1), br], -2)
        Ascale = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, H, -1, -1)
        A = Atrans @ Arot @ Askew @ Ascale @ Arot_pre
        return A.reshape(B * H, 3, 3)

    def gen_quaternion_batch(self, gaze_vec_batch):
        nx, ny, nz = gaze_vec_batch.unbind(dim=1)
        phi = torch.arctan2(ny, nx)  # Azimuthal angle 
        theta = torch.arccos(nz)     # Rotation angle
        # Compute quaternion components
        w = torch.cos(theta / 2)
        x = -torch.sin(phi) * torch.sin(theta / 2)
        y = torch.cos(phi) * torch.sin(theta / 2)
        z = torch.zeros_like(w)
        q_batch = torch.stack((w, x, y, z), dim=1)
        return q_batch

    def gen_geometric_correction_matrix(self, Q_batch, P_batch):
        '''
        Perform geometric correction of the 3D points using the provided quaternion
        :param Q_batch: A batch of quaternions for the rotation
        :param P_batch: A batch of 3D points to be rotated
        '''
        w, x, y, z = Q_batch.unbind(dim=1)
        convert_matrix = torch.stack([
            torch.stack([1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=1),
            torch.stack([2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w], dim=1),
            # torch.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2], dim=1)
        ], dim=1)   # Shape: [batch, 2, 3]
        corr_coor_batch = torch.matmul(convert_matrix, P_batch)
        return corr_coor_batch
    
    @staticmethod
    def ncc_batch(I_batch, J_all):
        uI = I_batch - I_batch.mean(dim=(-2, -1), keepdim=True)
        uJ = J_all - J_all.mean(dim=(-2, -1), keepdim=True)
        nom = torch.einsum('bshw,srhw->bsr', uI, uJ)
        denom = torch.norm(uI, dim=(-2, -1)).unsqueeze(-1) * torch.norm(uJ, dim=(-2, -1))  # [bs,1] * [sr]
        r = torch.nan_to_num(nom / denom)
        return r
    
    @staticmethod
    def fast_ncc_batch(I_batch, J_all, eps=1e-8):   #this code is slower
    # Subtract mean (along H,W)
        I_norm = I_batch - I_batch.mean(dim=(-2, -1), keepdim=True)  # [B, S, H, W]
        J_norm = J_all - J_all.mean(dim=(-2, -1), keepdim=True)      # [S, R, H, W]
                # Normalize: now each patch has unit norm
        I_norm = F.normalize(I_norm.flatten(start_dim=-2), dim=-1)  # [B, S, H*W]
        J_norm = F.normalize(J_norm.flatten(start_dim=-2), dim=-1)  # [S, R, H*W]
        r = torch.einsum('bsi,sri->bsr', I_norm, J_norm)  # [B, S, R]
        # denom = torch.norm(uI, dim=(-2, -1)).unsqueeze(-1) * torch.norm(uJ, dim=(-2, -1))  # [bs,1] * [sr]
        # r = torch.nan_to_num(nom / denom)                            # [B, S, R]
        return r

    @staticmethod
    def gen_polar_tensor_coord(el, B, H, W, device):
        epsilon = 0.1
        r_inner = el['pupil_radius'].unsqueeze(1)                  # [B, 1]
        r_outer = el['iris_radius'].unsqueeze(1) * (1 - epsilon)   # [B, 1]

        # Radial steps (0 to 1), interpolate between r_inner and r_outer
        r_stp = torch.linspace(0, 1, H, device=device)             # [H]
        r_mesh = r_inner + (r_outer - r_inner) * r_stp             # [B, H]
        r_mesh = r_mesh.unsqueeze(2).expand(-1, -1, W)             # [B, H, W]

        # Angular steps from -pi to pi (W samples)
        t_step = torch.linspace(-torch.pi, torch.pi, W, device=device)  # [W]
        t_mesh = t_step.view(1, 1, W).expand(B, H, W)                   # [B, H, W]
        return r_mesh, t_mesh
    

    def sample_iris_batch(self, frame_batch):
        polar_time00 = time.time()
        B, H, W = self.batch_size, self.img_h_pxl, self.img_w_pxl
        device = self.device

        el = frame_batch['ellipses']
        img_useful_batch = frame_batch['useful_maps']
        # unwrapped iris tensor coordinates:
        r_mesh, t_mesh = TorsionTracker.gen_polar_tensor_coord(el, B, H, W, device)
        
        # Intermidiate tensor coordinates (normalzied iris ring)
        Z = torch.polar(r_mesh.flatten(), t_mesh.flatten())  # [B*H*W]
        X_norm, Y_norm = Z.real.view(1, -1), Z.imag.view(1, -1)
        if self.params['torsion_geometric_correction_type']=='3D' and self.params['do_gaze_tracking']:
            R_eye = self.canonical_eyeball_radius   #R_eye doesn't affect the result, but only the scale
            Z_norm = torch.sqrt(R_eye**2 - X_norm**2 - Y_norm**2)
            P = torch.cat((X_norm, Y_norm, Z_norm), dim=0).float()
            np_per_batch = P.shape[1] //  B
            P_batch = P.unfold(1, np_per_batch, np_per_batch).permute(1, 0, 2)  # Shape: [B, 3, W*H]
            df_gaze = pd.DataFrame(frame_batch['gaze'])
            # unprojected_pupil_centers = np.vstack(df_gaze['3D_pupil_center'].values)
            gaze_vec_batch = torch.from_numpy(np.vstack(df_gaze['3D_pupil_norm'].values).astype(np.float32)).to(device)
            Q_batch = self.gen_quaternion_batch(gaze_vec_batch)
            P_corr_batch = self.gen_geometric_correction_matrix(Q_batch, P_batch)
            pupil_center_batch = torch.stack((el['pupil_center_x'],el['pupil_center_y']), axis=-1)
            shift_batch = (torch.nanmean(P_corr_batch, axis=-1) - pupil_center_batch)
            P_corr_batch = P_corr_batch - shift_batch.unsqueeze(2)
        else:
            Z_norm = torch.ones_like(X_norm).float()
            P = torch.cat((X_norm, Y_norm, Z_norm), dim=0)
            np_per_batch = P.shape[1] // B
            P_batch = P.unfold(1, np_per_batch, np_per_batch).permute(1, 0, 2)  # Shape: [B, 3, W*H]
            if self.params['torsion_geometric_correction_type']=='2D':
                A_torch_batch = TorsionTracker.ellipse2affine_batch(el, device)
                P_corr_batch = torch.bmm(A_torch_batch, P_batch)   #transform coord to ellipse coord
            elif self.params['torsion_geometric_correction_type']=='polish_2D':
                # ellipse params
                A_torch_batch = TorsionTracker.polish_el2affine_batch(el, B, H, device)
                P = P_batch.view(B, 3, H, W).permute(0, 2, 1, 3).reshape(B*H, 3, W)
                P_corr = torch.bmm(A_torch_batch, P)  # (B*H, 3, W)
                P_corr_batch = P_corr.view(B, H, 3, W).permute(0, 2, 1, 3).reshape(B, 3, -1)
                        
        P_corr_batch_normalized = P_corr_batch[:,:2,:].clone()
        P_corr_batch_normalized[:, 0, :] = 2*(P_corr_batch[:, 0, :]/(self.vid_w - 1))- 1 
        P_corr_batch_normalized[:, 1, :] = 2*(P_corr_batch[:, 1, :]/(self.vid_h - 1))- 1 
        # # Perform grid sampling
        # expects grid to be [N, H_out, W_out, 2], where last dim corresponds to (x, y) coordinate
        grid = P_corr_batch_normalized[:, :2, :].permute(0, 2, 1).view(B, H, W, 2)
        img_useful_batch_4d = img_useful_batch.unsqueeze(1)  # requirement of input shape [N, C, H, W]
        #mapping values from img_useful_batch to corresponding polar coord defined by grid
        iris_sampled_img_batch = F.grid_sample(img_useful_batch_4d, grid, mode='bilinear', align_corners=True)   #mode='bicubic'

        #takes expreminly long time for mps, main contribution of image optimization, imporve accuracy drastically 
        # TODO: Image intensity enhancement: computationally expensive (sobel_filter/F_vison.equalize)
        ROI_subimgs_batch = (iris_sampled_img_batch > 0.01).squeeze()  
        iris_sampled_img_batch = kornia_enhance.equalize_clahe(torch.nan_to_num(iris_sampled_img_batch, nan=0.0)).squeeze()   
        # iris_sampled_img_batch = iris_sampled_img_batch.squeeze()
        # iris_sampled_img_batch = self.sobel_filter(iris_sampled_img_batch.unsqueeze(1)).squeeze()
        # aaa = F_vison.equalize((iris_sampled_img_batch* 255).to(torch.uint8).unsqueeze(1)).squeeze().to(torch.float32)/255
        # from skimage.exposure import equalize_adapthist as adhist
        # test = adhist(iris_sampled_img_batch[0,:,:].cpu().numpy())
        if ROI_subimgs_batch.dim() == 2:
            ROI_subimgs_batch = ROI_subimgs_batch.unsqueeze(0)
        iris_sampled_img_batch = iris_sampled_img_batch*ROI_subimgs_batch
        iris_sampled_img_batch, success = self.normalizeRobust_batch(iris_sampled_img_batch,percentiles=[5, 95], bg_threshold=0.05)

        # ROI_subimgs_batch = iris_sampled_img_batch > 0.1
        img_useful_batch_nan = img_useful_batch.clone()
        seg_useful_batch = img_useful_batch_nan > 0   #TODO: Need to check if this is correct!
        img_useful_batch_nan[~seg_useful_batch] = torch.nan
        raw_img_brightness = torch.nanmean(img_useful_batch_nan, dim = [-2,-1])
    
        subimg_shape = iris_sampled_img_batch.shape
        ROI_cover_rate_batch = torch.sum(ROI_subimgs_batch,[-2,-1])/(subimg_shape[-2]*subimg_shape[-1])
        # quality_measures = dict({'polar_coverage_percent': ROI_cover_rate_batch})
        xgrad_batch = torch.abs(iris_sampled_img_batch[:,:,1:] - iris_sampled_img_batch[:,:,:-1])
        masked_xgrad_batch = xgrad_batch * ROI_subimgs_batch[:,:,1:]
        xgrad_sum_batch = torch.sum(masked_xgrad_batch, [-2,-1])
        ROI_sum_batch = torch.sum(ROI_subimgs_batch[:,:,1:], [-2,-1])
        xgrad_average_batch = xgrad_sum_batch / ROI_sum_batch

        quality_measures = dict({'polar_coverage_percent': ROI_cover_rate_batch,
                        'polar_xgradient_sum': xgrad_sum_batch,
                        'polar_xgradient_average': xgrad_average_batch,
                        'pattern_brightness':raw_img_brightness})

        # self.is_valid = self.is_valid & (ROI_cover_rate_batch>0.4).cpu().numpy()    #blink detection
        
        # subimg_height = int(np.round(iris_sampled_img_batch.shape[1]/self.n_subimags))
        # extra_hight = self.n_subimags*subimg_height - iris_sampled_img_batch.shape[1]

        # if extra_hight>0:
        #     zeros_tensor = torch.zeros(batch_size, extra_hight, iris_sampled_img_batch.shape[2], device=self.device)
        #     # Padding zeros tensor to the original tensor along the y-axis (dim=1)
        #     iris_sampled_img_batch = torch.cat((iris_sampled_img_batch, zeros_tensor), dim=1)
        # else:
        #     iris_sampled_img_batch = iris_sampled_img_batch[:, :self.n_subimags*subimg_height, :]
        
        # iris_sampled_subimgs_batch = iris_sampled_img_batch.reshape(batch_size, self.n_subimags, subimg_height, iris_sampled_img_batch.shape[2])
        
        ## Add noize at non-iris pattern erea -> the performance get worse!! (because of Gaussan????)
        # gaussian_values = torch.randn_like(iris_sampled_subimgs_batch)
        # iris_sampled_subimgs_batch[~ROI_subimgs_batch] = gaussian_values[~ROI_subimgs_batch]

        # return iris_sampled_img_batch, quality_measures, success
        polar_time01 = time.time()
        self.polar_elapsed_time += polar_time01 - polar_time00
        return iris_sampled_img_batch, quality_measures, success

    def indices_to_flat(self, row_indices, col_indices, cols):
        return row_indices * cols + col_indices

    def find_torsion_batch(self, batch_imgs, sample_img, algorithm = 'stochastic'):
        TM_time00 = time.time()
        device = self.device
        pad_pxl = self.pad_tempW_pxl
        batch_size = batch_imgs.shape[0]

        if algorithm == 'stochastic':
            # Assuming current_iris_img is already on the GPU and has size [50, 3600]
            n_samples = self.n_subimgs   # Number of sub-images to sample
            subimg_w_temp = self.subimg_w_pxl + 2 * pad_pxl
            # Compute valid start range for rows and columns to ensure sub-images fit within the original image
            h_lims = [int(np.ceil(self.subimg_h_pxl//2)),int(np.floor(self.img_h_pxl - (self.subimg_h_pxl // 2)))]
            w_lims = [0, self.img_w_pxl]
            # # Randomly sample middle point (x-y corrdination) of sub-image
            h_mid = torch.randint(h_lims[0], h_lims[1], (n_samples,), device=device)
            w_mid = torch.randint(w_lims[0], w_lims[1], (n_samples,), device=device)
            # # Generating all indices for each sub-image
            h_all = h_mid[:, None] + torch.arange(-self.subimg_h_pxl // 2, self.subimg_h_pxl // 2, device=device)
            w_all = (w_mid[:, None] + torch.arange(-subimg_w_temp // 2, subimg_w_temp // 2, device=device)) % self.img_w_pxl
            tempimg_flat = sample_img.view(-1)  # Flatten the image
            temp_subimgs_idx_map = self.indices_to_flat(h_all[:,:,None], w_all[:,None,:], self.img_w_pxl)
            tempsubimg_flat = tempimg_flat[temp_subimgs_idx_map.view(-1)]  # Gather pixels
            subtemp_imgs = tempsubimg_flat.view(temp_subimgs_idx_map.shape)  # Reshape to sub-images
            fix_patches_slided_all = subtemp_imgs.unfold(2, self.subimg_w_pxl, 1).permute(0, 2, 1, 3) #torch.Size([30, 101, 15, 200])
        
            #make mov_patches_batch
            curr_subimgs_idx_map = self.indices_to_flat(h_all[:,:,None],w_all[:,None,pad_pxl:-pad_pxl],self.img_w_pxl)
            curr_subimgs_idx_map_batch = curr_subimgs_idx_map.unsqueeze(0).repeat(batch_size, 1, 1, 1)
            # Compute flattened indices for the entire batch
            batch_offsets = (torch.arange(batch_size) * sample_img.numel()).view(-1, 1, 1, 1).to(self.device)
            flattened_indices = curr_subimgs_idx_map_batch + batch_offsets
            # Flatten the entire image batch
            curimg_flat = batch_imgs.view(-1)
            # Gather pixels using the computed flat indices and reshape
            cursubimg_flat = curimg_flat[flattened_indices.view(-1)]
            mov_patches_batch = cursubimg_flat.view(curr_subimgs_idx_map_batch.shape)

        if algorithm == 'subimage':
            subimg_h = int(np.round(self.img_h_pxl/self.n_subimgs))
            extra_h = self.n_subimgs*subimg_h - self.img_h_pxl
            if extra_h>=0:
                zeros_tensor = torch.zeros(batch_size, extra_h, self.img_w_pxl, device=device)
                zeros_tensor_samp = torch.zeros(extra_h, self.img_w_pxl, device=device)
                # Padding zeros tensor to the original tensor along the y-axis (dim=1)
                batch_imgs = torch.cat((batch_imgs, zeros_tensor), dim=-2)
                sample_img = torch.cat((sample_img, zeros_tensor_samp), dim=-2)
            else:
                batch_imgs = batch_imgs[:, :self.n_subimgs*subimg_h, :]
                sample_img = sample_img[:, :self.n_subimgs*subimg_h, :]
                extra_h = self.n_subimgs*subimg_h - batch_imgs.shape[1]

            mov_patches_batch = batch_imgs.reshape(batch_size, self.n_subimgs, subimg_h, self.img_w_pxl)
            temp_subimgs = sample_img.reshape(self.n_subimgs, subimg_h, self.img_w_pxl)
            subtemp_imgs = torch.concatenate((temp_subimgs[:,:,temp_subimgs.shape[-1]-pad_pxl:], temp_subimgs, temp_subimgs[:,:,0:pad_pxl]),axis =-1)
            fix_patches_slided_all = subtemp_imgs.unfold(2, pad_pxl*2+1, 1).permute(0, 3, 1, 2)

        # "fixed" patches sampled from template
        # "moving" patches sampled from current iris pattern
        # Calculate NCC for all samples and shifts in parallel
        # mov_patches = sub_current_images   #torch.Size([Sample, h, w])
        # fix_patches_slided_all = self.template_iris_img.shape   #torch.Size([Sample, pad_ix*2+1, h, w])
        # compiled_ncc = torch.compile(self.ncc_batch)   #should do it in the initialization
        # cc_batch = TorsionTracker.ncc_batch(mov_patches_batch, fix_patches_slided_all)
        cc_batch = TorsionTracker.ncc_batch(mov_patches_batch, fix_patches_slided_all)
        best_shift_idx_batch = torch.argmax(cc_batch, dim=-1)
        angular_shifts = torch.linspace(-pad_pxl*self.angular_pxl2deg,
                                    pad_pxl*self.angular_pxl2deg,2*pad_pxl+1,device=device)
        angular_shifts = angular_shifts.unsqueeze(0).repeat(cc_batch.shape[0], 1)
        ccs_batch = angular_shifts[torch.arange(angular_shifts.shape[0]).unsqueeze(1), best_shift_idx_batch]
        
        # Remove NaNs
        # valid_temp_shifts_batch = torch.where(torch.isnan(temp_shifts_batch), torch.tensor(float('nan'), device=temp_shifts_batch.device), temp_shifts_batch)
        # Calculate the number of valid (non-NaN) values
        # # # Calculate the 25th and 75th percentiles for each batch
        # Calculate the 25th and 75th percentiles along the specified dimension
        ROI_subimgs_batch = mov_patches_batch > 0.01
        ROI_sampimg = subtemp_imgs > 0.01
        ROI_sampimg_batch_expand = ROI_sampimg.unsqueeze(0).repeat(batch_size, 1, 1, 1)
        ROI_batch = ROI_subimgs_batch & ROI_sampimg_batch_expand[:,:,:,pad_pxl:-pad_pxl]
        ROI_shape = ROI_batch.shape
        ROI_coverrate_batch = torch.nansum(ROI_batch,[-2,-1])/(ROI_shape[-2]*ROI_shape[-1])  #batch_size x n_sample
        ROI_coverrate_batch = (ROI_coverrate_batch > self.cover_threshold) * ROI_coverrate_batch
        weight = (ROI_coverrate_batch/torch.nansum(ROI_coverrate_batch, dim=1, keepdim=True))
        
        # torsion_angle_out_batch = torch.nansum(ccs_batch*weight, dim = 1)   #Expectation value
        cc_batch[ccs_batch*weight == 0] = torch.nan
        torsion_angle_out_batch, _ = torch.nanmedian(ccs_batch, dim = 1)   #Expectation value
        # torsion_var_batch = torch.var(ccs_batch, dim = 1)
        
        # import matplotlib
        # matplotlib.use('Agg')
        # plt.figure(figsize=(6, 3))
        # def normalize(x):
        #     return (x - x.min()) / (x.max() - x.min())
        # # X-axis
        # x = np.linspace(-pad_pxl * self.angular_pxl2deg, pad_pxl * self.angular_pxl2deg, 2 * pad_pxl + 1)
        # # Extract and normalize NCC curves
        # ncc_curves = [
        #     normalize(cc_batch[0][0].cpu().numpy()),
        #     normalize(cc_batch[0][4].cpu().numpy()),
        #     normalize(cc_batch[0][10].cpu().numpy()),
        # ]
        # for i, curve in enumerate(ncc_curves):
        #     plt.plot(x, curve, label=f'patch {i}')

        # max_positions = [x[np.argmax(curve)] for curve in ncc_curves]
        # median_max_x = np.median(max_positions)
        # plt.axvline(median_max_x, color='red', linestyle='--', linewidth=2, label='median peak')
        # plt.xlabel('Angular Shift (degrees)')
        # plt.ylabel('NCC')
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(r"C:\Users\jzhao\Downloads\ccs_plot.pdf")  # Save to disk
        # plt.close()

        TM_time01 = time.time()
        self.TM_elapsed_time += TM_time01 -TM_time00
        return torsion_angle_out_batch, ccs_batch
    
    
    def update_template(self, iris_sampled_img, frame_idx, quality_measures, template_torsion_offset=0.0):
        self.template_iris_img  = iris_sampled_img
        self.template_frame_idx = frame_idx
        self.template_quality   = quality_measures
        self.template_torsion_offset = template_torsion_offset
        if not self.template_available:
            # this is set to True with the very first valid iris pattern
            self.template_available = True
            
    
    def torsion_tracker(self, frame_batch):
        time00 = time.time()
        self.batch_size = frame_batch['idxs'].shape[0]
        self.is_valid = frame_batch['is_valid']

        iris_sampled_batch, quality_measures_batch, success_batch = self.sample_iris_batch(frame_batch)
        good_coverage = quality_measures_batch['polar_coverage_percent']>0.7
        good_pattern_brightness = (quality_measures_batch['pattern_brightness'] > self.params['th_under_exposure']) & (quality_measures_batch['pattern_brightness'] < self.params['th_over_exposure'])
        no_blink = (frame_batch['ellipses']['pupil_confidence'] > self.params['threshold_confidence_pupil'])
        # torsion_val = no_blink & success_batch & good_coverage
        torsion_val_frames = self.is_valid & success_batch
        torsion_val_update = good_coverage & good_pattern_brightness & no_blink
        
        # Create/update template
        defalt_ix = np.where(torsion_val_frames)[0]
        # if not(self.template_available) and defalt_ix.numel() != 0:    #set default template
        if np.any(torsion_val_frames):
            if not(self.template_available):
                ix = defalt_ix[0]
                quality_measures_temp = dict({'polar_coverage_percent': quality_measures_batch['polar_coverage_percent'][ix],
                                    'polar_xgradient_sum': quality_measures_batch['polar_xgradient_sum'][ix],
                                    'polar_xgradient_average': quality_measures_batch['polar_xgradient_average'][ix]})

                self.update_template(iris_sampled_batch[ix,...],
                                    frame_batch['idxs'][ix],
                                    quality_measures_temp,
                                    template_torsion_offset= 0.0)

            #if self.template is exist, execute following code
            if self.template_available:
                torsion_angles_batch_relative, _ = self.find_torsion_batch(batch_imgs=iris_sampled_batch,
                                                                sample_img=self.template_iris_img,
                                                                algorithm = self.TM_algorithm)
                torsion_angles_batch = self.template_torsion_offset + torsion_angles_batch_relative
                torsion_angles_batch[~torsion_val_frames] = torch.nan    #if False, torsion_angle = 0.0
                
                # if frame_batch['idxs'][0] < self.torsion_update_interval:
                quality1 = quality_measures_batch['polar_coverage_percent']
                quality2 = quality_measures_batch['polar_xgradient_sum']
                quality_temp = (quality1>self.template_quality['polar_coverage_percent']) \
                    & (quality2>self.template_quality['polar_xgradient_sum']) & torsion_val_frames & torsion_val_update

            if (any(frame_batch['idxs'] % self.update_interval == 0) and (frame_batch['idxs'][0]!=0)):
                defalt_ix = np.where(torsion_val_frames)[0]
                ix = defalt_ix[0]
                quality_measures_temp = dict({'polar_coverage_percent': quality_measures_batch['polar_coverage_percent'][ix],
                                    'polar_xgradient_sum': quality_measures_batch['polar_xgradient_sum'][ix],
                                    'polar_xgradient_average': quality_measures_batch['polar_xgradient_average'][ix]})
                self.update_template(iris_sampled_batch[ix,...],
                                    frame_batch['idxs'][ix],
                                    quality_measures_temp,
                                    template_torsion_offset= torsion_angles_batch[ix].item())
                
                better_ix = torch.where(quality_temp)[0]
                if better_ix.numel() != 0:
                    quality1_condition = quality1[better_ix].unsqueeze(1) > quality1[better_ix].unsqueeze(0)
                    quality2_condition = quality2[better_ix].unsqueeze(1) > quality2[better_ix].unsqueeze(0)
                    combined_condition = quality1_condition & quality2_condition
                    ix = better_ix[torch.argmax(combined_condition.sum(dim=1))]
                    quality_measures_temp = dict({'polar_coverage_percent': quality_measures_batch['polar_coverage_percent'][ix],
                                        'polar_xgradient_sum': quality_measures_batch['polar_xgradient_sum'][ix],
                                        'polar_xgradient_average': quality_measures_batch['polar_xgradient_average'][ix]})

                    self.update_template(iris_sampled_batch[ix,...],
                                        frame_batch['idxs'][ix],
                                        quality_measures_temp,
                                        template_torsion_offset=torsion_angles_batch[ix].item())
            
        else:
            torsion_angles_batch = torch.full((self.batch_size,),torch.nan, dtype=torch.float32, device = self.device)

            # #physiological boundary
            # if  abs(self.template_torsion_offset)>15 and \
            #     any(frame_batch['idxs'] % self.torsion_update_interval == 0) and (frame_batch['idxs'][0]!=0):     

        frame_batch['is_valid'] = self.is_valid
        time01 = time.time()
        self.elapsed_time += (time01 - time00)
        
        frame_batch_torsion_viz = None
        # if self.params['viz_torsion'] or self.params['torsion_collecte_detail']:
        #     frame_batch_torsion_viz = {'idxs': frame_batch['idxs'],
        #                             'torsion_angles': torsion_angles_batch,
        #                             # 'torsion_distribution': torsion_distri_batch,
        #                             'template_iris_imgs': self.template_iris_img,
        #                             'current_iris_imgs': iris_sampled_batch}
            
        if self.use_queue:
            self.threads['ques']['torsion_out'].put(
                frame_batch_torsion_viz if self.params['torsion_collecte_detail'] else torsion_angles_batch
            )
            # if self.params['viz_torsion']:
            #     self.threads['ques']['torsion_visualization'].put(frame_batch_torsion_viz)

        else:
            torsion_angles_batch
            # return frame_batch_torsion_viz if self.params['viz_torsion'] or self.params['torsion_collecte_detail'] else torsion_angles_batch