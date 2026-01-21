import ast
import os
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
from ..utils.torsion_process import clean_signal, interpolate_nan
from vispy import scene
from vispy.visuals.transforms import STTransform
from vispy.visuals.filters.clipping_planes import PlanesClipper
from vispy.scene.cameras import TurntableCamera
from numpy.lib.stride_tricks import sliding_window_view
from matplotlib import cm


def draw_ellipse(image, el, color, t, scale=1.0):
    theta, cx, cy, a, b = el
    cv2.ellipse(
        image,
        (int(cx + 0.5), int(cy + 0.5)),
        (int(a * scale + 0.5), int(b * scale + 0.5)),
        float(np.degrees(theta)),
        0, 360,
        color,
        t + 1,
        lineType=cv2.LINE_AA,
    )

    
def gen_sphere_mesh(r, num_grids=25, insert_theta=None):
    theta_vals = np.linspace(0, np.pi, num_grids)
    if insert_theta is not None:
        theta_vals = np.sort(np.append(theta_vals, insert_theta))
    phi_vals = np.linspace(0, 2 * np.pi, num_grids)
    T, U = np.meshgrid(theta_vals, phi_vals, indexing='ij')
    X = r * np.sin(T) * np.cos(U)
    Y = r * np.sin(T) * np.sin(U)
    Z = r * np.cos(T)
    return np.array([X, Y, Z])

def rot_to_align_with_z(v):
    v = np.asarray(v, dtype=float)
    v = v / np.linalg.norm(v)
    z = np.array([0.0, 0.0, 1.0])
    # Already aligned with +z
    if np.allclose(v, z):
        return np.eye(3)
    # Aligned with -z → 180° flip around any axis ⟂ z (e.g. x-axis)
    if np.allclose(v, -z):
        return Quaternion_Rotation.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0])).as_matrix()

    # General case
    axis = np.cross(v, z)
    axis /= np.linalg.norm(axis)
    cos_angle = np.clip(np.dot(v, z), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    return Quaternion_Rotation.from_rotvec(axis * angle).as_matrix()

def fit_legrand_model(eyeball_params, df_gaze_tp, torsion_tp, large_meshes, small_meshes, num_grids=25, use_mask=True):
    r_e, r_c, r_s = eyeball_params['eyeball_radius'], eyeball_params['corneaball_radius'], eyeball_params['limbus_radius']
    d_p = np.sqrt(r_e**2 - r_s**2)
    d_ec = d_p - np.sqrt(r_c**2 - r_s**2)
    r_p = df_gaze_tp.circle_3d['radius']
    c_eye = np.asarray(df_gaze_tp.sphere['center'])
    c_pupil = np.asarray(df_gaze_tp.circle_3d['center'])
    gaze_vector_raw = c_pupil - c_eye   
    norm = np.linalg.norm(gaze_vector_raw)  #same as df_gaze_tp.circle_3d['normal']
    if norm < 1e-6 or np.isnan(c_eye).any() or np.isnan(c_pupil).any(): return None
    gaze_vec = gaze_vector_raw / norm
    c_cornea = c_eye + d_ec * gaze_vec
    c_pupil_unc = c_eye + d_p * gaze_vec
    gaze_align_rot = rot_to_align_with_z(gaze_vec)
    R_gaze = gaze_align_rot.T
    R_torsion = Quaternion_Rotation.from_rotvec(torsion_tp * gaze_vec).as_matrix()
    R_total = R_torsion @ R_gaze
    # torsion_rot    = Quaternion_Rotation.from_rotvec(torsion_tp * gaze_vec).as_matrix()
    # gaze_align_rot = Quaternion_Rotation.align_vectors(gaze_vec[None], [[0, 0, 1]])[0].as_matrix()
    # torsion_rot = Quaternion_Rotation.from_rotvec(torsion_tp * gaze_vec).as_matrix()
    # R_total = torsion_rot @ gaze_align_rot

    def rotate_and_shift(coords, center, rot_matrix): return np.einsum('ij,jkl->ikl', rot_matrix, coords) + center[:, None, None]
    large_meshes = rotate_and_shift(large_meshes, c_eye, R_total)
    small_meshes = rotate_and_shift(small_meshes, c_cornea, R_total)
    if use_mask:
        small_mask = np.sum((small_meshes - c_eye[:, None, None])**2, axis=0) > (r_e**2 - 1e-1)
        large_mask = np.sum((large_meshes - c_cornea[:, None, None])**2, axis=0) < (r_c**2 - 1e-1)
        large_coords_masked = np.where(large_mask, np.nan, large_meshes)
        small_coords_masked = np.where(small_mask, small_meshes, np.nan)
    else:
        large_coords_masked, small_coords_masked = large_meshes, small_meshes
    theta = np.linspace(0, 2*np.pi, num_grids)
    perp_axis1 = np.cross(gaze_vec, [1, 0, 0])
    if np.linalg.norm(perp_axis1) < 1e-6: perp_axis1 = np.cross(gaze_vec, [0, 1, 0])
    perp_axis1 /= np.linalg.norm(perp_axis1)
    perp_axis2 = np.cross(gaze_vec, perp_axis1)
    perp1_rot, perp2_rot = R_torsion @ perp_axis1, R_torsion @ perp_axis2
    circle_basis = np.outer(np.cos(theta), perp1_rot) + np.outer(np.sin(theta), perp2_rot)
    limbus_circle_points = c_pupil_unc[:, None] + r_s * circle_basis.T
    pupil_circle_points = c_pupil_unc[:, None] + r_p * circle_basis.T

    theta = np.deg2rad(df_gaze_tp['ellipse']['angle'])
    cx, cy = df_gaze_tp['ellipse']['center']
    a, b = df_gaze_tp['ellipse']['axes']
    entpup_el = np.array([theta, cx, cy, a/2, b/2], dtype=float)

    return {
        "eyeball_mesh": large_coords_masked, 
        "corneaball_mesh": small_coords_masked,
        "limbus_circle_points": limbus_circle_points, 
        "pupil_circle_points": pupil_circle_points,
        "pupil_center": c_pupil, 
        "pupil_center_unc": c_pupil_unc,
        "entpup_el": entpup_el,
        "gaze_vec": gaze_vec, 
        "eyeball_center": c_eye, 
        "corneaball_center": c_cornea
    }

def project_3d_to_2d(X, Y, Z, camera_params):
    vid_w, vid_h = camera_params['resolution']
    fcl_mm = camera_params['fcl_mm']
    scaling_factor = camera_params['mm2px_scaling']
    return (
        fcl_mm * (X / Z) * scaling_factor + vid_w * 0.5,
        fcl_mm * (Y / Z) * scaling_factor + vid_h * 0.5,
    )

class SegFitVisualizer:
    def __init__(self, camera_params, eyeball_params, frame_size, height=200, color_map=None):
        self.frame_h, self.frame_w = frame_size
        self.height = height
        self.camera_params = camera_params
        self.eyeball_params = eyeball_params
        self.color_map = color_map or {
                                        "eyeball_mesh": (255, 150, 50), "corneaball_mesh": (255, 150, 255),
                                        "limbus_circle_points": (255, 0, 255), "pupil_circle_points": (255, 255, 0),
                                        "eyeball_center": (0, 0, 255), "pupil_center": (0, 255, 127), 
                                        "gaze_vector": (0, 255, 127),
                                        "entpup_el": (144, 238, 144),
                                    }
        self.labels = ["Original", "Overlay", "Fitted"]
        self.font_scale = 0.5
        self.font = cv2.FONT_HERSHEY_DUPLEX

    def _to_bgr(self, img):
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img

    def _add_title_bar(self, img, text, text_h=30):
        h, w = img.shape[:2]
        bar = np.full((text_h, w, 3), 255, np.uint8)
        cv2.putText(bar,text,
        (10, int(text_h * 0.75)),
        self.font, self.font_scale,  # e.g., 0.6 or 0.8
        (0, 0, 0),
        1,  # thinner text
        lineType=cv2.LINE_AA)
        return np.vstack([bar, img])

    def _project_3d_to_2d(self, X, Y, Z):
        w, h = self.camera_params['resolution']
        fcl, scale = self.camera_params['fcl_mm'], self.camera_params['mm2px_scaling']
        return (fcl * (X / Z) * scale + w * 0.5, fcl * (Y / Z) * scale + h * 0.5)

    def _generate_overlay(self, frame, el_info, sclera_mask):
        iris, pupil = [
            ((int(el_info[cx]), int(el_info[cy])), (int(el_info[w]), int(el_info[h])), el_info[r] * 180 / np.pi)
            for cx, cy, w, h, r in [
                ("iris_center_x", "iris_center_y", "iris_w", "iris_h", "iris_radian"),
                ("pupil_center_x", "pupil_center_y", "pupil_w", "pupil_h", "pupil_radian"),
            ]
        ]
        masks = {k: np.zeros_like(frame[:, :, 0], dtype=np.uint8) for k in ["iris", "pupil"]}
        for key, params in zip(masks, [iris, pupil]):
            cv2.ellipse(masks[key], *params, 0, 360, 1, -1)
        ring_mask = (((masks["iris"] == 1) & (masks["pupil"] == 0)) & (sclera_mask > 0)).astype(np.uint8) * 255
        return cv2.addWeighted(frame, 1.0,
                               np.where(ring_mask[..., None] == 255, self.color_map["limbus_circle_points"], 0).astype(np.uint8),
                               0.3, 0)

    def _render_model_fitting(self, result, image):
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

        if "gaze_vec" in result:
            end = (int(pupil2d[0] + 50 * result["gaze_vec"][0]), int(pupil2d[1] + 50 * result["gaze_vec"][1]))
            cv2.line(image, tuple(map(int, pupil2d)), end, self.color_map["gaze_vector"], t * 2, lineType=cv2.LINE_AA)
        return image

    def render(self, frame, el_info, sclera_mask, result):
        overlay = self._generate_overlay(frame, el_info, sclera_mask)
        fitted = self._render_model_fitting(result, frame.copy())
        panels = [frame, overlay, fitted]
        labeled_panels = []
        for img, label in zip(panels, self.labels):
            img_bgr = self._to_bgr(img)
            labeled_img = self._add_title_bar(img_bgr, label, text_h=int(self.frame_h // 8))
            labeled_panels.append(labeled_img)
        labeled_panels = np.vstack(labeled_panels)
        aspect_ratio = labeled_panels.shape[1]/labeled_panels.shape[0]
        resized_panels = cv2.resize(labeled_panels, (int(self.height * aspect_ratio), self.height))
        return resized_panels


class VispyEyeballRenderer:
    def __init__(self, initial_center, r_e=12.0, r_c=7.8, r_s=6.0, res=30, width=400, height=500):
        self.r_e, self.r_c, self.r_s = r_e, r_c, r_s
        self.d_p = np.sqrt(r_e**2 - r_s**2)
        self.d_ec = self.d_p - np.sqrt(r_c**2 - r_s**2)
        self.width, self.height = width, height
        self._camera_initialized = False
        self._camera_center = np.array(initial_center, dtype=np.float32)
        self.limbus_pairs = [(0, 12), (4, 16), (8, 20)]

# initialize will affect by which one is called first (last one will be showed in the front)
        self.canvas = scene.SceneCanvas(size=(self.width, self.height), bgcolor='white', show=False)
        self.canvas.app.process_events()   # ensure context is created
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = TurntableCamera(fov=50, azimuth=0, elevation=-90, up='+z')
        self.view.camera.center = tuple(self._camera_center)
        extent = 10
        self.view.camera.set_range(
            x=[self._camera_center[0] - extent + 8, self._camera_center[0] + extent],
            y=[self._camera_center[1] - extent + 1.5, self._camera_center[1] + extent],
            z=[self._camera_center[2] - extent - 1, self._camera_center[2] + extent]
        )
        # Create visuals
        self.eyeball = scene.visuals.Sphere(radius=r_e, rows=res, cols=res, method='latitude',
                                            color=(0.8, 0.8, 0.8, 1.0), parent=self.view.scene)
        
        self.pupil_line = scene.visuals.Line(color='black', width=2, method='gl', parent=self.view.scene)
        self.limbus_dots = scene.visuals.Markers(parent=self.view.scene)
        self.limbus_connect_lines = [
            scene.visuals.Line(color='red', width=3, parent=self.view.scene),
            scene.visuals.Line(color='green', width=3, parent=self.view.scene),
            scene.visuals.Line(color='blue', width=3, parent=self.view.scene)
        ]

        # Transparent cornea (alpha = 0.5) drawn last to prevent z-fighting
        self.cornea = scene.visuals.Sphere(radius=r_c, rows=res, cols=res, method='latitude',
                                           color=(1.0, 0.7, 1.0, 1.0), parent=self.view.scene)

        self.arrow_size = 10
        self.arrow = scene.visuals.Arrow(
            arrows=np.array([[0, 0, 0], [1, 1, 1]]).reshape((1, 2, 3)),  # start to end
            arrow_size= self.width//100,
            width= 5,
            arrow_color= (0, 0, 0, 1.0),
            color=(0, 0, 0, 1.0),         # Set shaft (line) color
            parent=self.view.scene,
        )
        self.clipper = PlanesClipper(coord_system='scene')
        self.eyeball.attach(self.clipper)
        self.eyeball.transform = STTransform()
        self.cornea.transform = STTransform()
        self.pupil_loop = np.array([[i, (i + 1) % 25] for i in range(25)], dtype=np.int32)

    def render(self, result):
        gaze = result['gaze_vec'] / np.linalg.norm(result['gaze_vec'])
        pupil = result['pupil_center']
        limbus_points = result['limbus_circle_points']
        cornea_center = result['eyeball_center'] + self.d_ec * gaze
        limbus_center = result['eyeball_center'] + self.d_p * gaze
        self.eyeball.transform.translate = result['eyeball_center']
        self.cornea.transform.translate = cornea_center
        self.clipper.clipping_planes = np.array([[limbus_center, -gaze]], dtype=np.float32)

        self.pupil_line.set_data(pos=result['pupil_circle_points'].T.astype(np.float32),
                                 connect=self.pupil_loop)
        self.limbus_dots.set_data(pos=limbus_points[:,0::4].T.astype(np.float32),
                                  face_color='blue', edge_color='black', size=8)
        
        self.arrow.set_data(pos=np.array([pupil, pupil + self.arrow_size * gaze], dtype=np.float32))
        self.arrow.set_data(arrows=np.array([[pupil, pupil + self.arrow_size * gaze]], dtype=np.float32))
        self.arrow.set_gl_state(depth_test=False)
        # self.arrow.set_gl_state('translucent', depth_test=True)
        # Draw lines between opposite limbus points
        for i, (idx1, idx2) in enumerate(self.limbus_pairs):
            line_pts = limbus_points[:, [idx1, idx2]].T.astype(np.float32)
            self.limbus_connect_lines[i].set_data(pos=line_pts)

        img = self.canvas.render()
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        # return cv2.resize(img, (int(img.shape[1] * self.height / img.shape[0]), self.height))
        return img
    


class EyeMovementVisualizer:
    def __init__(self, processed_signal, width=500, height=480, max_frame_show=500, colors=None):
        self.width, self.height = width, height
        # self.plot_width = self.width - 50
        self.max_frame_show = max_frame_show
        self.y_axis_end = int((self.height // 3)* 0.5)
        self.colors = colors or {
            'horizontal': (255, 0, 0),
            'vertical': (0, 0, 255),
            'torsion': (0, 255, 0)
        }
        self.font = cv2.FONT_HERSHEY_DUPLEX
        self.font_scale = 0.5
        self.font_scale_title = 0.7
        self.digit_w = cv2.getTextSize(f"0", cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)[0][0]   # (width, height) letter
        self.axis_x_start =  self.digit_w*5 + self.width//30
        self.axis_x_end = self.width - self.width//15
        self.x_frame2pxl = (self.axis_x_end - self.axis_x_start) / max_frame_show
        self.tick_start = self.axis_x_start + self.axis_x_start//8
        self.tick_frame = np.linspace(0, self.max_frame_show - self.max_frame_show//10, 5, dtype=int)
        self.tick_pxl = (self.tick_start + self.tick_frame * self.x_frame2pxl).astype(np.int32)

        self.signals_clean = {k: interpolate_nan(np.asarray(v, np.float32)) for k, v in processed_signal.items()}
        self.signal_stats = self._calc_stats()
        self.xs_map = self._precompute_x_coords()
        self.y_offsets = {k: i *self.height // 3 + int(self.height // 3*0.3) for i, k in enumerate(['horizontal', 'vertical', 'torsion'])}
        self.bg_template = self._create_static_background()

    def _calc_stats(self):
        stats = {}
        for k, x in self.signals_clean.items():
            x_pad = np.pad(x, (self.max_frame_show - 1, 0), mode='edge')
            win = sliding_window_view(x_pad, self.max_frame_show)
            stats[k] = {'min': win.min(1), 'ptp': win.ptp(1) + 1e-6}
        return stats
    
    def _precompute_x_coords(self):
        x = np.linspace(self.axis_x_start, self.axis_x_end, self.max_frame_show).astype(np.int32).reshape(-1, 1, 1)
        return {l: x[:l] for l in range(2, self.max_frame_show + 1)}

    def _create_static_background(self):
        bg = np.full((self.height, self.width, 3), 255, np.uint8)
        # Y-axis lines and labels
        for key, y_axis_start in self.y_offsets.items():
            label = key.title()   #Horizontal, Vertical, Torsion
            size = cv2.getTextSize(label, self.font, self.font_scale_title, 1)[0]  #(width, height) letter
            cx = self.axis_x_start + int((self.axis_x_end - self.axis_x_start)/2 - size[0]/2)  # Center the label horizontally
            cv2.putText(bg, label, (cx, y_axis_start - int(size[1])), 
                        self.font, self.font_scale_title, self.colors[key], 1, lineType=cv2.LINE_AA)  #put label (horizontal, vertical, torsion)
            cv2.line(bg, (self.axis_x_start, y_axis_start),    #start of Y-axis line (x, y coordinates)
                     (self.axis_x_start, y_axis_start + self.y_axis_end),   #end of Y-axis line (x, y coordinates)
                     (0, 0, 0), 1)    # Y-axis line
        # Only tick **lines** here — not labels
        for i in self.tick_pxl:
            cv2.line(bg, (i, self.height - 30), (i, self.height - 20), (0, 0, 0), 1)
        return bg

    def render(self, tp):
        bg = self.bg_template.copy()
        # Update frame number
        size = cv2.getTextSize(f"Frame: {tp}", cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1)[0]   # (width, height) letter
        cv2.putText(bg, f"Frame: {tp}", 
                    (self.width - size[0] - self.digit_w, self.y_offsets['horizontal']- int(self.height // 3*0.15)), 
                    self.font, self.font_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        for key in ['horizontal', 'vertical', 'torsion']:
            y0 = self.y_offsets[key]
            data = self.signals_clean[key][max(0, tp - self.max_frame_show):tp]
            if len(data) < 2:
                continue
            min_v = self.signal_stats[key]['min'][tp - 1]
            ptp_v = self.signal_stats[key]['ptp'][tp - 1]
            norm = (data - min_v) / ptp_v
            ys = (y0 + self.y_axis_end * (1 - norm)).astype(np.int32)
            xs = self.xs_map[len(data)].squeeze()
            pts = np.stack((xs, ys), axis=-1).reshape(-1, 1, 2)
            cv2.polylines(bg, [pts], False, self.colors[key], 1)

            # Draw y axis digit points
            max_ylabel, min_ylabel = f"{np.max(data):.1f}", f"{np.min(data):.1f}"
            cv2.putText(bg, max_ylabel, (self.width//30 + self.digit_w*(5-len(max_ylabel)), y0 + 10), self.font, self.font_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)
            cv2.putText(bg, min_ylabel, (self.width//30 + self.digit_w*(5-len(min_ylabel)), y0 + self.y_axis_end), self.font, self.font_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

            if np.min(data) <= 0 <= np.max(data):
                zero_y = int(y0 + self.y_axis_end * (1 - (0 - min_v) / ptp_v))
                cv2.line(bg, (self.axis_x_start, zero_y), (self.axis_x_end, zero_y), (150, 150, 150), 1, lineType=cv2.LINE_AA)
                # cv2.putText(bg, "0", (self.axis_x + 5, zero_y - 2), self.font, self.font_scale, (0, 0, 0), 1)
            
                for i, frame in enumerate(self.tick_frame):
                    if tp >= self.max_frame_show:
                        tick_value = tp - self.max_frame_show + frame if frame < self.max_frame_show else tp + frame - self.max_frame_show
                    else:
                        tick_value = frame
                    digit_position = self.tick_pxl[i] - self.digit_w*len(str(tick_value))//2
                    cv2.putText(bg, str(tick_value), (digit_position, self.height - 5), self.font, self.font_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)
        return bg
    


from matplotlib import cm
def stack_polar_maps_colored(polar_current, polar_TM, output_size=(600, 200)):
    """
    Stack two polar maps vertically with matplotlib colormap and titles added after resizing.

    Args:
        polar_current (np.ndarray): Top image.
        polar_TM (np.ndarray): Bottom image.
        output_size (tuple): (width, height) of the full output panel.

    Returns:
        np.ndarray: Final stacked and titled panel (BGR uint8).
    """
    def to_colored(img):
        norm = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)
        colored = cm.viridis(norm)[:, :, :3]
        return (colored * 255).astype(np.uint8)

    def add_title(image, title, bar_height=30):
        h, w = image.shape[:2]
        bar = np.full((bar_height, w, 3), 255, np.uint8)
        font = cv2.FONT_HERSHEY_DUPLEX
        scale, thickness = 0.6, 1
        text_size = cv2.getTextSize(title, font, scale, thickness)[0]
        pos = ((w - text_size[0]) // 2, (bar_height + text_size[1]) // 2)
        cv2.putText(bar, title, pos, font, scale, (0, 0, 0), thickness, cv2.LINE_AA)
        return np.vstack([bar, image])

    # Calculate target size for each image (half of full height)
    target_w, total_h = output_size
    target_h = total_h // 2

    img1 = cv2.resize(to_colored(polar_current), (target_w, target_h), interpolation=cv2.INTER_AREA)
    img2 = cv2.resize(to_colored(polar_TM), (target_w, target_h), interpolation=cv2.INTER_AREA)

    img1 = add_title(img1, "Current Iris Map")
    img2 = add_title(img2, "Template Iris Map")

    combined = np.vstack([img1, img2])
    return combined
