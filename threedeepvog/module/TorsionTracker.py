# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import os, torch, time, cv2, threading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import kornia.enhance as kornia_enhance


class TorsionTracker(threading.Thread):
    """
    Thread-TorsionTracker

    Purpose
    -------
    Estimate ocular torsion (rotation of the iris texture around the gaze axis) over time.

    Pipeline (high level)
    ---------------------
    1) Build a polar “iris strip” image per frame:
       - Use pupil/iris geometry (ellipses) to define an annulus.
       - Sample the annulus into a polar map (radial x angular) via torch.grid_sample.
       - Optionally apply geometric correction:
           * 2D ellipse-based affine (“2D”)
           * row-wise (radial) affine interpolation (“polish_2D”)
           * 3D correction using gaze direction (“3D”)

    2) Enhance / normalize polar map:
       - Mask invalid pixels (background / missing iris).
       - CLAHE (kornia) to boost local contrast.
       - Robust normalization using percentiles (nanquantile).

    3) Template matching on the polar map to estimate torsion:
       - Maintain a "template" polar map (best quality so far).
       - Compare current polar map to template using NCC (normalized cross-correlation).
       - Two modes:
           * "stochastic": sample random sub-patches; evaluate many angular shifts.
           * "subimage": split map into sub-bands; evaluate shifts.

       Output torsion angle = template_offset + estimated_relative_shift.

    4) Template update:
       - Periodically (update_interval) and/or when quality improves:
         update template_iris_img + template_torsion_offset.

    Outputs
    -------
    - torsion_angles_batch: torch.Tensor [B] (degrees) with NaNs for invalid frames.
    - Sent to threads['ques']['torsion_out'] if use_queue=True.

    Notes / Caveats
    ---------------
    - This is memory-heavy: NCC compares many patches x many shifts.
      Large pad_tempW_pxl, many samples, or large patch sizes can OOM on GPU.
    - Some parts of the pipeline are faster on CPU depending on hardware / PyTorch build.
    - For stability: many operations use NaNs to ignore invalid pixels.
    """

    def __init__(self, threads, args, daemon=False, use_queue=True):
        super().__init__(daemon=daemon)
        self.name = 'Thread-TorsionTracker'
        self.threads = threads
        self.args = args
        self.use_queue = args.get('is_parallel', use_queue)  # prefer args['is_parallel'] when present
        self.frame_counter = -1

        # -----------------------
        # Device / video geometry
        # -----------------------
        self.device = args['device']             # torch.device
        self.W = args['vid_w']                   # processed width
        self.H = args['vid_h']                   # processed height

        # -----------------------
        # Torsion configuration
        # -----------------------
        self.update_interval = int(args['torsion_force_update_template_interval_sec'] * args['vid_fps'])

        # Angular sampling resolution in the polar map
        self.angular_pxl2deg = args['torsion_angular_pxl2deg']         # degrees per pixel (angular)
        self.img_w_pxl = int(360 / self.angular_pxl2deg)               # polar map width (angles)
        args['torsion_angular_pxl'] = self.img_w_pxl

        # Radial sampling resolution in the polar map
        self.img_h_pxl = args['torsion_radial_pxl']                    # polar map height (radius)
        args['torsion_polarmap_h_pxl'] = self.img_h_pxl
        args['torsion_polarmap_w_pxl'] = self.img_w_pxl

        # Template matching setup
        self.TM_algorithm = args['torsion_TM_algorithm']               # "stochastic" or "subimage"
        self.n_subimgs = args['torsion_n_subimgs']                     # number of sub-patches
        self.cover_threshold = args['torsion_coverage_rate_threshold'] # reject patches with low overlap
        self.pad_tempW_pxl = int(np.ceil(args['torsion_max_deg'] / self.angular_pxl2deg))  # max shift padding (px)
        self.subimg_w_pxl = int(np.ceil(args['torsion_subimg_angular_deg'] / self.angular_pxl2deg))  # patch width (px)
        self.subimg_h_pxl = args['torsion_subimg_h_pxl']               # patch height (px)

        self.unpdate_flag = False

        # -----------------------
        # Timing / profiling
        # -----------------------
        self.elapsed_time = 0
        self.polar_elapsed_time = 0
        self.TM_elapsed_time = 0

        # -----------------------
        # Geometry scale (mm -> px)
        # -----------------------
        self.mm2px = np.linalg.norm(np.array(self.args['resolution'])) / np.linalg.norm(np.array(self.args['sensor_size']))
        self.r_eye_canonical = 12.0 * self.mm2px   # canonical eyeball radius in pixels (used in 3D correction)

        # -----------------------
        # Template state
        # -----------------------
        self.template_available = False
        self.template_iris_img = torch.zeros((self.img_h_pxl, self.img_w_pxl),
                                             dtype=torch.float32, device=self.device)
        self.template_frame_idx = -1

        # Quality metrics used to decide template updates
        self.template_quality = {
            'polar_coverage_percent': -1.0,
            'polar_xgradient_sum': -1.0,
            'polar_xgradient_average': -1.0,
        }

        # Torsion offsets: absolute torsion = template_offset + relative_shift
        self.template_torsion_offset = np.nan

        # Optional storage for later debugging/analysis
        self.torsion_angles = []
        self.current_iris_imgs = []

        # -----------------------
        # If using 3D geometric correction, load eyeball model
        # -----------------------
        if self.args['torsion_geometric_correction_type'] == '3D':
            eyeball_info = pd.read_json(self.args['eyeball_path'], orient='index').T
            self.c_eye = eyeball_info.loc[0, ['eye_centre_x', 'eye_centre_y', 'eye_centre_z']].values.reshape(3, 1)
            self.r_eye = eyeball_info.loc[0, 'aver_eye_radius']

    def normalizeRobust_batch(self, img_batch, percentiles=[5, 95], bg_threshold=0.05):
        """
        Robust intensity normalization on a batch of polar maps.

        - Treat pixels <= bg_threshold as invalid (background).
        - Compute per-image percentiles over valid pixels (nanquantile).
        - Rescale to [0,1] and clamp.
        - Returns success mask: (th_high - th_low) > 0.
        """
        if img_batch.dim() == 2:
            img_batch = img_batch.unsqueeze(0)

        q = torch.tensor(percentiles, dtype=torch.float32, device=self.device) / 100.0
        valid_pxl_mask = img_batch > bg_threshold

        valid_pxls = torch.where(valid_pxl_mask, img_batch, torch.tensor(torch.nan, device=self.device))
        th_low, th_high = torch.nanquantile(valid_pxls.view(img_batch.shape[0], -1), q, dim=-1, keepdim=True).squeeze()

        # If no valid pixels, nanquantile can yield nan; guard those
        th_low[th_low == torch.nan] = 0
        th_high[th_high == torch.nan] = 1

        img_batch -= th_low.reshape(-1, 1, 1)
        img_batch /= (th_high.reshape(-1, 1, 1) - th_low.reshape(-1, 1, 1))
        img_batch = torch.clip(img_batch, 0.0, 1.0)

        success_batch = ((th_high - th_low) > 0).squeeze()
        img_batch = torch.where(success_batch.view(-1, 1, 1), img_batch, torch.zeros_like(img_batch))
        return img_batch, success_batch

    @staticmethod
    def ellipse2affine_batch(el, device):
        """
        Build a per-frame affine matrix (3x3) for ellipse geometric correction (2D).
        Uses pupil ellipse parameters to build: translate * rotate * skew * scale * inverse-rotate.
        """
        theta, w, h, loc_x, loc_y = (
            el['pupil_radian'],
            el['pupil_w'],
            el['pupil_h'],
            el['pupil_center_x'],
            el['pupil_center_y'],
        )
        skew_x = 2 * w / (w + h)
        skew_y = 2 * h / (w + h)

        Arot_pre_batch = torch.stack([
            torch.stack([torch.cos(-theta), -torch.sin(-theta), torch.zeros_like(theta)], dim=1),
            torch.stack([torch.sin(-theta),  torch.cos(-theta), torch.zeros_like(theta)], dim=1),
            torch.tensor([0, 0, 1], dtype=torch.float32, device=device).repeat(theta.size(0), 1),
        ], dim=1)

        Arot_batch = torch.stack([
            torch.stack([torch.cos(theta), -torch.sin(theta), torch.zeros_like(theta)], dim=1),
            torch.stack([torch.sin(theta),  torch.cos(theta), torch.zeros_like(theta)], dim=1),
            torch.tensor([0, 0, 1], dtype=torch.float32, device=device).repeat(theta.size(0), 1),
        ], dim=1)

        Askew_batch = torch.stack([
            torch.stack([skew_x, torch.zeros_like(skew_x), torch.zeros_like(skew_x)], dim=1),
            torch.stack([torch.zeros_like(skew_y), skew_y, torch.zeros_like(skew_y)], dim=1),
            torch.tensor([0, 0, 1], dtype=torch.float32, device=device).repeat(theta.size(0), 1),
        ], dim=1)

        Ascale_batch = torch.eye(3, dtype=torch.float32, device=device).repeat(theta.size(0), 1, 1)

        Atrans_batch = torch.stack([
            torch.stack([torch.ones_like(loc_x), torch.zeros_like(loc_x), loc_x], dim=1),
            torch.stack([torch.zeros_like(loc_y), torch.ones_like(loc_y), loc_y], dim=1),
            torch.tensor([0, 0, 1], dtype=torch.float32, device=device).repeat(theta.size(0), 1),
        ], dim=1)

        A_batch = torch.bmm(torch.bmm(torch.bmm(torch.bmm(Atrans_batch, Arot_batch), Askew_batch), Ascale_batch), Arot_pre_batch)
        return A_batch

    @staticmethod
    def polish_el2affine_batch(el, B, H, device):
        """
        "polish_2D" correction: per-radial-row affine transforms that smoothly interpolate between pupil and iris ellipses.
        Returns [B*H,3,3] transforms, applied row-wise.
        """
        theta = el['pupil_radian'].view(B, 1)
        epsilon = 0.1
        stp = torch.linspace(0, 1, H, device=device)

        sx_pup = (2 * el['pupil_w'] / (el['pupil_w'] + el['pupil_h'])).unsqueeze(1)
        sy_pup = (2 * el['pupil_h'] / (el['pupil_w'] + el['pupil_h'])).unsqueeze(1)
        sx_iris = (2 * el['iris_w'] / (el['iris_w'] + el['iris_h'])).unsqueeze(1)
        sy_iris = (2 * el['iris_h'] / (el['iris_w'] + el['iris_h'])).unsqueeze(1)

        cx_pup, cy_pup = el['pupil_center_x'].unsqueeze(1), el['pupil_center_y'].unsqueeze(1)
        cx_iris, cy_iris = el['iris_center_x'].unsqueeze(1), el['iris_center_y'].unsqueeze(1)

        cx_delta, cy_delta = (cx_iris - cx_pup) * (1 - epsilon), (cy_iris - cy_pup) * (1 - epsilon)
        sx_delta, sy_delta = (sx_iris - sx_pup) * (1 - epsilon), (sy_iris - sy_pup) * (1 - epsilon)

        x_shift = (cx_pup + cx_delta * stp)
        y_shift = (cy_pup + cy_delta * stp)
        x_skew = (sx_pup + sx_delta * stp)
        y_skew = (sy_pup + sy_delta * stp)

        cos_t = torch.cos(theta).expand(B, H)
        sin_t = torch.sin(theta).expand(B, H)
        z = torch.zeros(B, H, device=device)
        o = torch.ones(B, H, device=device)
        br = torch.tensor([0, 0, 1], device=device).repeat(B * H, 1).view(B, H, 3)

        Arot = torch.stack([torch.stack([cos_t, -sin_t, z], -1), torch.stack([sin_t, cos_t, z], -1), br], -2)
        Arot_pre = torch.stack([torch.stack([cos_t, sin_t, z], -1), torch.stack([-sin_t, cos_t, z], -1), br], -2)
        Askew = torch.stack([torch.stack([x_skew, z, z], -1), torch.stack([z, y_skew, z], -1), br], -2)
        Atrans = torch.stack([torch.stack([o, z, x_shift], -1), torch.stack([z, o, y_shift], -1), br], -2)
        Ascale = torch.eye(3, device=device).view(1, 1, 3, 3).expand(B, H, -1, -1)

        A = Atrans @ Arot @ Askew @ Ascale @ Arot_pre
        return A.reshape(B * H, 3, 3)

    def gen_quaternion_batch(self, gaze_vec_batch):
        """
        Convert gaze vector to a simplified quaternion used by the 3D correction code.
        (Assumes torsion around z is 0; used to rotate canonical iris texture.)
        """
        nx, ny, nz = gaze_vec_batch.unbind(dim=1)
        phi = torch.arctan2(ny, nx)
        theta = torch.arccos(nz)

        w = torch.cos(theta / 2)
        x = -torch.sin(phi) * torch.sin(theta / 2)
        y = torch.cos(phi) * torch.sin(theta / 2)
        z = torch.zeros_like(w)
        return torch.stack((w, x, y, z), dim=1)

    def gen_geometric_correction_matrix(self, Q_batch, P_batch):
        """
        Apply a quaternion-derived 2x3 rotation matrix to 3D points P_batch.
        Returns corrected coordinates in 2D (x,y), still in pixel-like space after later shifting.
        """
        w, x, y, z = Q_batch.unbind(dim=1)
        convert_matrix = torch.stack([
            torch.stack([1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w], dim=1),
            torch.stack([2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w], dim=1),
        ], dim=1)  # [B,2,3]
        return torch.matmul(convert_matrix, P_batch)

    @staticmethod
    def ncc_batch(I_batch, J_all):
        """
        Normalized cross-correlation (NCC) between:
          I_batch: [B, S, H, W] moving patches
          J_all:   [S, R, H, W] fixed patches for each shift

        Returns:
          r: [B, S, R] NCC score per (batch, patch, shift)
        """
        uI = I_batch - I_batch.mean(dim=(-2, -1), keepdim=True)
        uJ = J_all - J_all.mean(dim=(-2, -1), keepdim=True)
        nom = torch.einsum('bshw,srhw->bsr', uI, uJ)
        denom = torch.norm(uI, dim=(-2, -1)).unsqueeze(-1) * torch.norm(uJ, dim=(-2, -1))
        return torch.nan_to_num(nom / denom)

    @staticmethod
    def gen_polar_tensor_coord(el, B, H, W, device):
        """
        Build polar sampling coordinates:
          r_mesh: [B,H,W] radial values between pupil and iris radii
          t_mesh: [B,H,W] angle values from -pi..pi
        """
        epsilon = 0.1
        r_inner = el['pupil_radius'].unsqueeze(1)
        r_outer = el['iris_radius'].unsqueeze(1) * (1 - epsilon)

        r_stp = torch.linspace(0, 1, H, device=device)
        r_mesh = (r_inner + (r_outer - r_inner) * r_stp).unsqueeze(2).expand(-1, -1, W)

        t_step = torch.linspace(-torch.pi, torch.pi, W, device=device)
        t_mesh = t_step.view(1, 1, W).expand(B, H, W)
        return r_mesh, t_mesh

    def sample_iris_batch(self, frame_batch):
        """
        Main polar-map builder.

        Inputs:
        - frame_batch['useful_maps']: iris texture map in image coordinates
        - frame_batch['ellipses']: pupil/iris params
        - optionally frame_batch['gaze'] if doing 3D correction

        Outputs:
        - iris_sampled_img_batch: [B, H_polar, W_polar] float32 in [0,1]
        - quality_measures: dict of per-frame quality signals
        - success: [B] boolean for normalization success
        """
        polar_time00 = time.time()
        B, H, W = self.batch_size, self.img_h_pxl, self.img_w_pxl
        device = self.device

        el = frame_batch['ellipses']
        img_useful_batch = frame_batch['useful_maps']

        r_mesh, t_mesh = TorsionTracker.gen_polar_tensor_coord(el, B, H, W, device)
        Z = torch.polar(r_mesh.flatten(), t_mesh.flatten())
        X_norm, Y_norm = Z.real.view(1, -1), Z.imag.view(1, -1)

        # ---- geometric correction ----
        if self.args['torsion_geometric_correction_type'] == '3D' and self.args['gaze_tracking_flag']:
            R_eye = self.r_eye_canonical
            Z_norm = torch.sqrt(R_eye**2 - X_norm**2 - Y_norm**2)

            P = torch.cat((X_norm, Y_norm, Z_norm), dim=0).float()
            np_per_batch = P.shape[1] // B
            P_batch = P.unfold(1, np_per_batch, np_per_batch).permute(1, 0, 2)  # [B,3,W*H]

            df_gaze = pd.DataFrame(frame_batch['gaze'])
            gaze_vec_batch = torch.from_numpy(np.vstack(df_gaze['3D_pupil_norm'].values).astype(np.float32)).to(device)

            Q_batch = self.gen_quaternion_batch(gaze_vec_batch)
            P_corr_batch = self.gen_geometric_correction_matrix(Q_batch, P_batch)

            pupil_center_batch = torch.stack((el['pupil_center_x'], el['pupil_center_y']), axis=-1)
            shift_batch = (torch.nanmean(P_corr_batch, axis=-1) - pupil_center_batch)
            P_corr_batch = P_corr_batch - shift_batch.unsqueeze(2)
        else:
            Z_norm = torch.ones_like(X_norm).float()
            P = torch.cat((X_norm, Y_norm, Z_norm), dim=0)
            np_per_batch = P.shape[1] // B
            P_batch = P.unfold(1, np_per_batch, np_per_batch).permute(1, 0, 2)

            if self.args['torsion_geometric_correction_type'] == '2D':
                A_torch_batch = TorsionTracker.ellipse2affine_batch(el, device)
                P_corr_batch = torch.bmm(A_torch_batch, P_batch)
            elif self.args['torsion_geometric_correction_type'] == 'polish_2D':
                A_torch_batch = TorsionTracker.polish_el2affine_batch(el, B, H, device)
                P_ = P_batch.view(B, 3, H, W).permute(0, 2, 1, 3).reshape(B * H, 3, W)
                P_corr = torch.bmm(A_torch_batch, P_)
                P_corr_batch = P_corr.view(B, H, 3, W).permute(0, 2, 1, 3).reshape(B, 3, -1)
            else:
                P_corr_batch = P_batch

        # Normalize to grid_sample coordinates [-1,1]
        P_corr_batch_normalized = P_corr_batch[:, :2, :].clone()
        P_corr_batch_normalized[:, 0, :] = 2 * (P_corr_batch[:, 0, :] / (self.W - 1)) - 1
        P_corr_batch_normalized[:, 1, :] = 2 * (P_corr_batch[:, 1, :] / (self.H - 1)) - 1

        grid = P_corr_batch_normalized.permute(0, 2, 1).view(B, H, W, 2)
        img_useful_batch_4d = img_useful_batch.unsqueeze(1)

        iris_sampled_img_batch = F.grid_sample(img_useful_batch_4d, grid, mode='bilinear', align_corners=True)

        # Contrast enhancement + robust normalization
        ROI_subimgs_batch = (iris_sampled_img_batch > 0.01).squeeze()
        iris_sampled_img_batch = kornia_enhance.equalize_clahe(torch.nan_to_num(iris_sampled_img_batch, nan=0.0)).squeeze()

        if ROI_subimgs_batch.dim() == 2:
            ROI_subimgs_batch = ROI_subimgs_batch.unsqueeze(0)

        iris_sampled_img_batch = iris_sampled_img_batch * ROI_subimgs_batch
        iris_sampled_img_batch, success = self.normalizeRobust_batch(
            iris_sampled_img_batch, percentiles=[5, 95], bg_threshold=0.05
        )

        # Quality metrics used for template update decisions
        img_useful_batch_nan = img_useful_batch.clone()
        seg_useful_batch = img_useful_batch_nan > 0
        img_useful_batch_nan[~seg_useful_batch] = torch.nan
        raw_img_brightness = torch.nanmean(img_useful_batch_nan, dim=[-2, -1])

        subimg_shape = iris_sampled_img_batch.shape
        ROI_cover_rate_batch = torch.sum(ROI_subimgs_batch, [-2, -1]) / (subimg_shape[-2] * subimg_shape[-1])

        xgrad_batch = torch.abs(iris_sampled_img_batch[:, :, 1:] - iris_sampled_img_batch[:, :, :-1])
        masked_xgrad_batch = xgrad_batch * ROI_subimgs_batch[:, :, 1:]
        xgrad_sum_batch = torch.sum(masked_xgrad_batch, [-2, -1])
        ROI_sum_batch = torch.sum(ROI_subimgs_batch[:, :, 1:], [-2, -1])
        xgrad_average_batch = xgrad_sum_batch / ROI_sum_batch

        quality_measures = {
            'polar_coverage_percent': ROI_cover_rate_batch,
            'polar_xgradient_sum': xgrad_sum_batch,
            'polar_xgradient_average': xgrad_average_batch,
            'pattern_brightness': raw_img_brightness,
        }

        self.polar_elapsed_time += (time.time() - polar_time00)
        return iris_sampled_img_batch, quality_measures, success

    def indices_to_flat(self, row_indices, col_indices, cols):
        """Convert 2D indices to flattened indices for fancy indexing."""
        return row_indices * cols + col_indices

    def find_torsion_batch(self, batch_imgs, sample_img, algorithm='stochastic'):
        """
        Estimate torsion by matching current polar maps to the template.

        batch_imgs: [B, H_polar, W_polar]
        sample_img: [H_polar, W_polar] template
        algorithm:  "stochastic" or "subimage"

        returns:
          torsion_angle_out_batch: [B] degrees
          ccs_batch:               [B, n_samples] per-sample best shifts (deg)
        """
        TM_time00 = time.time()
        device = self.device
        pad_pxl = self.pad_tempW_pxl
        batch_size = batch_imgs.shape[0]

        if algorithm == 'stochastic':
            n_samples = self.n_subimgs
            subimg_w_temp = self.subimg_w_pxl + 2 * pad_pxl

            h_lims = [int(np.ceil(self.subimg_h_pxl // 2)),
                      int(np.floor(self.img_h_pxl - (self.subimg_h_pxl // 2)))]
            w_lims = [0, self.img_w_pxl]

            h_mid = torch.randint(h_lims[0], h_lims[1], (n_samples,), device=device)
            w_mid = torch.randint(w_lims[0], w_lims[1], (n_samples,), device=device)

            h_all = h_mid[:, None] + torch.arange(-self.subimg_h_pxl // 2, self.subimg_h_pxl // 2, device=device)
            w_all = (w_mid[:, None] + torch.arange(-subimg_w_temp // 2, subimg_w_temp // 2, device=device)) % self.img_w_pxl

            tempimg_flat = sample_img.view(-1)
            temp_subimgs_idx_map = self.indices_to_flat(h_all[:, :, None], w_all[:, None, :], self.img_w_pxl)
            subtemp_imgs = tempimg_flat[temp_subimgs_idx_map.view(-1)].view(temp_subimgs_idx_map.shape)

            fix_patches_slided_all = subtemp_imgs.unfold(2, self.subimg_w_pxl, 1).permute(0, 2, 1, 3)

            curr_subimgs_idx_map = self.indices_to_flat(
                h_all[:, :, None], w_all[:, None, pad_pxl:-pad_pxl], self.img_w_pxl
            )
            curr_subimgs_idx_map_batch = curr_subimgs_idx_map.unsqueeze(0).repeat(batch_size, 1, 1, 1)

            batch_offsets = (torch.arange(batch_size) * sample_img.numel()).view(-1, 1, 1, 1).to(self.device)
            flattened_indices = curr_subimgs_idx_map_batch + batch_offsets

            curimg_flat = batch_imgs.view(-1)
            mov_patches_batch = curimg_flat[flattened_indices.view(-1)].view(curr_subimgs_idx_map_batch.shape)

        elif algorithm == 'subimage':
            subimg_h = int(np.round(self.img_h_pxl / self.n_subimgs))
            extra_h = self.n_subimgs * subimg_h - self.img_h_pxl

            if extra_h >= 0:
                batch_imgs = torch.cat((batch_imgs, torch.zeros(batch_size, extra_h, self.img_w_pxl, device=device)), dim=-2)
                sample_img = torch.cat((sample_img, torch.zeros(extra_h, self.img_w_pxl, device=device)), dim=-2)
            else:
                batch_imgs = batch_imgs[:, :self.n_subimgs * subimg_h, :]
                sample_img = sample_img[:self.n_subimgs * subimg_h, :]

            mov_patches_batch = batch_imgs.reshape(batch_size, self.n_subimgs, subimg_h, self.img_w_pxl)
            temp_subimgs = sample_img.reshape(self.n_subimgs, subimg_h, self.img_w_pxl)

            subtemp_imgs = torch.concatenate(
                (temp_subimgs[:, :, temp_subimgs.shape[-1] - pad_pxl:], temp_subimgs, temp_subimgs[:, :, 0:pad_pxl]),
                axis=-1
            )
            fix_patches_slided_all = subtemp_imgs.unfold(2, pad_pxl * 2 + 1, 1).permute(0, 3, 1, 2)

        else:
            raise ValueError(f"Unknown TM algorithm: {algorithm}")

        # NCC over all (sample, shift) combinations
        cc_batch = TorsionTracker.ncc_batch(mov_patches_batch, fix_patches_slided_all)
        best_shift_idx_batch = torch.argmax(cc_batch, dim=-1)

        angular_shifts = torch.linspace(
            -pad_pxl * self.angular_pxl2deg,
            pad_pxl * self.angular_pxl2deg,
            2 * pad_pxl + 1,
            device=device
        ).unsqueeze(0).repeat(cc_batch.shape[0], 1)

        ccs_batch = angular_shifts[torch.arange(angular_shifts.shape[0]).unsqueeze(1), best_shift_idx_batch]

        # Patch overlap weighting (reject low-coverage patches)
        ROI_subimgs_batch = mov_patches_batch > 0.01
        ROI_sampimg = subtemp_imgs > 0.01
        ROI_sampimg_batch_expand = ROI_sampimg.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        ROI_batch = ROI_subimgs_batch & ROI_sampimg_batch_expand[:, :, :, pad_pxl:-pad_pxl]
        ROI_shape = ROI_batch.shape
        ROI_coverrate_batch = torch.nansum(ROI_batch, [-2, -1]) / (ROI_shape[-2] * ROI_shape[-1])
        ROI_coverrate_batch = (ROI_coverrate_batch > self.cover_threshold) * ROI_coverrate_batch

        weight = (ROI_coverrate_batch / torch.nansum(ROI_coverrate_batch, dim=1, keepdim=True))

        cc_batch[ccs_batch * weight == 0] = torch.nan
        torsion_angle_out_batch, _ = torch.nanmedian(ccs_batch, dim=1)

        self.TM_elapsed_time += (time.time() - TM_time00)
        return torsion_angle_out_batch, ccs_batch

    def update_template(self, iris_sampled_img, frame_idx, quality_measures, template_torsion_offset=0.0):
        """Replace the current template with a higher-quality polar map (and store its torsion offset)."""
        self.template_iris_img = iris_sampled_img
        self.template_frame_idx = frame_idx
        self.template_quality = quality_measures
        self.template_torsion_offset = template_torsion_offset
        if not self.template_available:
            self.template_available = True

    def torsion_tracker(self, frame_batch):
        """
        Main torsion tracking step for a batch:
        - build polar maps
        - compute torsion vs template (if available)
        - optionally update template
        - push torsion output
        """
        time00 = time.time()
        self.batch_size = frame_batch['idxs'].shape[0]
        self.is_valid = frame_batch['is_valid']

        iris_sampled_batch, quality_measures_batch, success_batch = self.sample_iris_batch(frame_batch)

        good_coverage = quality_measures_batch['polar_coverage_percent'] > 0.7
        good_pattern_brightness = (
            (quality_measures_batch['pattern_brightness'] > self.args['th_under_exposure'])
            & (quality_measures_batch['pattern_brightness'] < self.args['th_over_exposure'])
        )
        no_blink = (frame_batch['ellipses']['pupil_confidence'] > self.args['threshold_confidence_pupil'])

        torsion_val_frames = self.is_valid & success_batch
        torsion_val_update = good_coverage & good_pattern_brightness & no_blink

        defalt_ix = np.where(torsion_val_frames)[0]

        if np.any(torsion_val_frames):
            # Default template if none exists
            if not self.template_available:
                ix = defalt_ix[0]
                quality_measures_temp = {
                    'polar_coverage_percent': quality_measures_batch['polar_coverage_percent'][ix],
                    'polar_xgradient_sum': quality_measures_batch['polar_xgradient_sum'][ix],
                    'polar_xgradient_average': quality_measures_batch['polar_xgradient_average'][ix],
                }
                self.update_template(
                    iris_sampled_batch[ix, ...],
                    frame_batch['idxs'][ix],
                    quality_measures_temp,
                    template_torsion_offset=0.0
                )

            # Estimate torsion relative to template
            torsion_angles_batch_relative, _ = self.find_torsion_batch(
                batch_imgs=iris_sampled_batch,
                sample_img=self.template_iris_img,
                algorithm=self.TM_algorithm
            )
            torsion_angles_batch = self.template_torsion_offset + torsion_angles_batch_relative
            torsion_angles_batch[~torsion_val_frames] = torch.nan

            # Template update decision
            quality1 = quality_measures_batch['polar_coverage_percent']
            quality2 = quality_measures_batch['polar_xgradient_sum']
            quality_temp = (
                (quality1 > self.template_quality['polar_coverage_percent'])
                & (quality2 > self.template_quality['polar_xgradient_sum'])
                & torsion_val_frames
                & torsion_val_update
            )

            # Forced periodic template refresh
            if (any(frame_batch['idxs'] % self.update_interval == 0) and (frame_batch['idxs'][0] != 0)):
                ix = defalt_ix[0]
                quality_measures_temp = {
                    'polar_coverage_percent': quality_measures_batch['polar_coverage_percent'][ix],
                    'polar_xgradient_sum': quality_measures_batch['polar_xgradient_sum'][ix],
                    'polar_xgradient_average': quality_measures_batch['polar_xgradient_average'][ix],
                }
                self.update_template(
                    iris_sampled_batch[ix, ...],
                    frame_batch['idxs'][ix],
                    quality_measures_temp,
                    template_torsion_offset=torsion_angles_batch[ix].item()
                )

                better_ix = torch.where(quality_temp)[0]
                if better_ix.numel() != 0:
                    quality1_condition = quality1[better_ix].unsqueeze(1) > quality1[better_ix].unsqueeze(0)
                    quality2_condition = quality2[better_ix].unsqueeze(1) > quality2[better_ix].unsqueeze(0)
                    combined_condition = quality1_condition & quality2_condition
                    ix = better_ix[torch.argmax(combined_condition.sum(dim=1))]

                    quality_measures_temp = {
                        'polar_coverage_percent': quality_measures_batch['polar_coverage_percent'][ix],
                        'polar_xgradient_sum': quality_measures_batch['polar_xgradient_sum'][ix],
                        'polar_xgradient_average': quality_measures_batch['polar_xgradient_average'][ix],
                    }
                    self.update_template(
                        iris_sampled_batch[ix, ...],
                        frame_batch['idxs'][ix],
                        quality_measures_temp,
                        template_torsion_offset=torsion_angles_batch[ix].item()
                    )
        else:
            torsion_angles_batch = torch.full((self.batch_size,), torch.nan, dtype=torch.float32, device=self.device)

        self.elapsed_time += (time.time() - time00)

        if self.use_queue:
            self.threads['ques']['torsion_out'].put(
                torsion_angles_batch if not self.args['torsion_collecte_detail'] else None
            )

    def run(self):
        """
        Thread main loop:
        - Receive frame batches from torsion_tracking queue.
        - Compute torsion under torch.no_grad().
        - Send torsion angles to torsion_out.
        - Exit cleanly on poison pill (None).
        """
        torch.set_grad_enabled(False)
        while True:
            frame_batch = self.threads['ques']['torsion_tracking'].get()
            if frame_batch is None:
                self.threads['ques']['torsion_out'].put(None)
                break
            with torch.no_grad():
                self.torsion_tracker(frame_batch)