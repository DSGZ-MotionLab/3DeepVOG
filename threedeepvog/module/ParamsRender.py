import threading
import queue
import cv2
import pandas as pd
import numpy as np
import torch
# from fast_deepvog3D.model3D.segmentation_model import SegResNet_3in3out_model
from scipy.spatial.transform import Rotation as Quaternion_Rotation
from ..utils.transformation import rend_params, to_numpy, to_torch, projection
from ..utils.visualization import draw_ellipse


class ParamsRender(threading.Thread):
    def __init__(self, threads, args, daemon=True, use_queue=True, maxsize=32):
        super().__init__(daemon=daemon)
        self.name = "Thread-ParamsRender"
        self.threads = threads
        self.args = args
        self.use_queue = use_queue
        self.q = queue.Queue(maxsize=maxsize)

        eyeball_info = pd.read_json(self.args['eyeball_path'],orient='index').T
        self.eye_centre = eyeball_info.loc[0, ['eye_centre_x', 'eye_centre_y', 'eye_centre_z']].values
        self.aver_eye_radius = eyeball_info.loc[0, 'aver_eye_radius']
        self.best_dice = eyeball_info.loc[0, 'best_dice']
        self.mm2px = np.linalg.norm(np.array(self.args['resolution'])) / np.linalg.norm(np.array(self.args['sensor_size']))
        self.num_grids = 25
        self.camera_cfg = {
            "img_size": (self.args["resolution"][1], self.args["resolution"][0]),   # (width, height)
            "fcl_mm": self.args["focal_length"],
            "mm2px": self.mm2px,
            "resolution": self.args["resolution"],
            "fcl_px": self.args["focal_length"] * self.mm2px,
        }

        self.color_map = self.args.get("color_map", {
            "eyeball_mesh": (255, 150, 50),
            "corneaball_mesh": (255, 150, 255),
            "iris_el": (255, 0, 255),
            "refpup_el": (255, 255, 0),
            "entpup_el": (144, 238, 144),
            "c_eye": (0, 0, 255),
            "c_pupil": (0, 255, 127),
            "gaze": (0, 255, 127),
        })
    
    def overlay_fit_model_cv(self, results: dict, images: np.ndarray) -> np.ndarray:
        """
        results: dict of batched arrays (B,...) produced by your pipeline.
                Must include:
                - "eyeball_mesh": (B, Hm, Wm, 3)
                - "corneaball_mesh": (B, Hm, Wm, 3)
                - "c_eye2d": (B, 2)
                - "c_pupil2d": (B, 2)
                - "entpup_el": (B, 5)  [theta_rad, cx, cy, a, b]
                - "refpup_el": (B, 5)  [theta_rad, cx, cy, a, b]   (optional)
                - "gaze_vec": (B, 3)   (optional)
        images: (B, H, W) uint8 grayscale (0..255)

        returns: (B, H, W, 3) uint8 BGR
        """
        B = images.shape[0]
        out = np.empty((B, images.shape[1], images.shape[2], 3), dtype=np.uint8)
        n = self.num_grids
        t = max(1, int(np.round((np.linalg.norm(self.args["resolution"]) / 500))))

        for ix in range(B):
            image = cv2.cvtColor(images[ix], cv2.COLOR_GRAY2BGR)

            # ---- draw meshes ----
            for mesh_key, color in (
                ("eyeball_mesh_2d", self.color_map["eyeball_mesh"]),
                ("corneaball_mesh_2d", self.color_map["corneaball_mesh"]),
            ):
                if mesh_key not in results:
                    continue
                xy = results[mesh_key]          # (B, Hm, Wm, 2)
                x, y = xy[ix,:,:, 0], xy[ix,:,:, 1]              # (Hm, Wm)
                # grid lines
                # NOTE: n should match your mesh grid size. If your mesh has different Hm/Wm,
                # you can set n = min(n, x.shape[0], x.shape[1]).
                nn = min(n, x.shape[0], x.shape[1])
                for i in range(nn):
                    for pts in (np.column_stack((x[i], y[i])),
                                np.column_stack((x[:, i], y[:, i]))):
                        pts = pts[~np.isnan(pts).any(axis=1)]
                        if pts.shape[0] > 1:
                            cv2.polylines(image, [pts.astype(np.int32)], False, color, t)

            # ---- keypoints ----
            c_eye2d = results["c_eye2d"][ix]
            c_pupil2d = results["c_pupil2d"][ix]
            cv2.circle(image, (int(round(c_eye2d[0])), int(round(c_eye2d[1]))), t + 2, self.color_map["c_eye"], -1)
            cv2.circle(image, (int(round(c_pupil2d[0])), int(round(c_pupil2d[1]))), t + 2, self.color_map["c_pupil"], -1)

            # ---- ellipses ----
            draw_ellipse(image, results["entpup_el"][ix], self.color_map["entpup_el"], t, scale=1.0)
            draw_ellipse(image, results["refpup_el"][ix], self.color_map["refpup_el"], t, scale=1.0) if "refpup_el" in results else None
            draw_ellipse(image, results["iris_el"][ix], self.color_map["iris_el"], t, scale=1.0) if "iris_el" in results else None  

            # ---- gaze vector ----
            if "gaze" in results:
                gv = results["gaze"][ix]
                end = (int(round(c_pupil2d[0] + 50 * gv[0])), int(round(c_pupil2d[1] + 50 * gv[1])))
                cv2.line(
                    image,(int(round(c_pupil2d[0])), int(round(c_pupil2d[1]))),end,
                    self.color_map["gaze"], t * 2, lineType=cv2.LINE_AA,
                )
            out[ix] = image
        return out


    def rendering(self, render_batch):
        frames_gray = render_batch['frame_gray']
        gaze_results = render_batch['gaze_results']
        processed_torsion = render_batch['torsion_results']
        B = frames_gray.shape[0]
        eyeball_cfg = {
            "r_eye": 12 * torch.ones(B, 1),   # in mm
            "r_cornea": 7.8 * torch.ones(B, 1),   # in mm
            "r_iris": 6.0 * torch.ones(B, 1),   # in mm,
            "c_eye": torch.from_numpy(self.eye_centre) * torch.ones((B, 3)),   # in mm,
            "num_grids": 25,
        }
        render_batch = {**gaze_results, 
                        **self.camera_cfg,
                        **eyeball_cfg,
                        "torsion": processed_torsion}
        out_dict = rend_params(to_torch(render_batch))
        fitted_frames = self.overlay_fit_model_cv(to_numpy(out_dict), frames_gray)
        if self.use_queue:
            self.threads['ques']['fitted_frame_out'].put(fitted_frames)


    def run(self):
        while True:
            render_batch = self.threads['ques']['params_rendering'].get()
            if render_batch is None:
                self.threads['ques']['fitted_frame_out'].put(None)
                break

            self.rendering(render_batch)