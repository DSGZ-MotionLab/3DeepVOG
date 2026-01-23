import threading
import queue
import cv2
import pandas as pd
import numpy as np
import torch
# from fast_deepvog3D.model3D.segmentation_model import SegResNet_3in3out_model
from scipy.spatial.transform import Rotation as Quaternion_Rotation
from ..utils.transformation import rend_params, to_numpy, to_torch
from ..utils.visualization import draw_ellipse


class ParamsRender(threading.Thread):
    def __init__(self, threads, args, daemon=True, use_queue=True, maxsize=32):
        super().__init__(daemon=daemon)
        self.name = "Thread-ParamsRender"
        self.threads = threads
        self.args = args
        self.use_queue = use_queue
        self.q = queue.Queue(maxsize=maxsize)

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
        B, H, W = images.shape
        out = np.empty((B, H, W, 3), dtype=np.uint8)

        t = max(1, int(np.round(np.linalg.norm(self.args["resolution"]) / 500)))
        nn = int(self.num_grids)

        mesh_specs = [
            ("eyeball_mesh_2d", self.color_map["eyeball_mesh"]),
            ("corneaball_mesh_2d", self.color_map["corneaball_mesh"]),
        ]
        xy_mesh = {k: results.get(k) for k, _ in mesh_specs}  # hoist once

        # ---- helpers ----
        def draw_point(img, xy, color, r):
            xy = np.asarray(xy, dtype=np.float32).reshape(-1)
            if xy.size < 2 or not np.isfinite(xy[:2]).all():
                return
            x, y = float(xy[0]), float(xy[1])
            if not (0 <= x < W and 0 <= y < H):
                return
            cv2.circle(img, (int(x + 0.5), int(y + 0.5)), r, color, -1, lineType=cv2.LINE_AA)

        def draw_line(img, p0, p1, color, thickness):
            p0 = np.asarray(p0, dtype=np.float32).reshape(-1)
            p1 = np.asarray(p1, dtype=np.float32).reshape(-1)
            if p0.size < 2 or p1.size < 2 or not (np.isfinite(p0[:2]).all() and np.isfinite(p1[:2]).all()):
                return
            x0, y0 = float(p0[0]), float(p0[1])
            x1, y1 = float(p1[0]), float(p1[1])
            if not ((-10 <= x0 <= W + 10 and -10 <= y0 <= H + 10) or (-10 <= x1 <= W + 10 and -10 <= y1 <= H + 10)):
                return
            cv2.line(
                img,
                (int(x0 + 0.5), int(y0 + 0.5)),
                (int(x1 + 0.5), int(y1 + 0.5)),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )

        # text style
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.3 if max(H, W) <= 400 else 0.4
        thick_txt = max(1, t)
        pad = 1

        # prefetch arrays (optional)
        hor_arr = results.get("hor", None)
        ver_arr = results.get("ver", None)

        for ix in range(B):
            img = cv2.cvtColor(images[ix], cv2.COLOR_GRAY2BGR)

            # ---- meshes ----
            for mesh_key, color in mesh_specs:
                xy = xy_mesh.get(mesh_key)
                if xy is None:
                    continue

                x = xy[ix, :, :, 0]
                y = xy[ix, :, :, 1]

                # IMPORTANT: your mesh is Hm x Wm; iterate both dims safely
                Hm, Wm = x.shape
                n_use = min(nn, Hm, Wm)

                for i in range(n_use):
                    for pts in (np.column_stack((x[i], y[i])), np.column_stack((x[:, i], y[:, i]))):
                        pts = pts[
                            np.isfinite(pts).all(axis=1)
                            & np.all(np.abs(pts) <= max(H, W) * 100, axis=1)
                        ]
                        if pts.shape[0] > 1:
                            # optional: clip before cast to avoid int overflow
                            pts[:, 0] = np.clip(pts[:, 0], -W * 10, W * 10)
                            pts[:, 1] = np.clip(pts[:, 1], -H * 10, H * 10)
                            cv2.polylines(img, [np.rint(pts).astype(np.int32)], False, color, t, lineType=cv2.LINE_AA)

            # ---- keypoints ----
            draw_point(img, results["c_eye2d"][ix],   self.color_map["c_eye"],   t + 2)
            draw_point(img, results["c_pupil2d"][ix], self.color_map["c_pupil"], t + 2)

            # ---- ellipses ----
            draw_ellipse(img, results["entpup_el"][ix], self.color_map["entpup_el"], t, scale=1.0)
            if "refpup_el" in results:
                draw_ellipse(img, results["refpup_el"][ix], self.color_map["refpup_el"], t, scale=1.0)
            if "iris_el" in results:
                draw_ellipse(img, results["iris_el"][ix], self.color_map["iris_el"], t, scale=1.0)

            # ---- gaze vector ----
            if "gaze" in results:
                gv = np.asarray(results["gaze"][ix], dtype=np.float32).reshape(-1)
                if gv.size >= 2 and np.isfinite(gv[:2]).all():
                    p0 = results["c_pupil2d"][ix]
                    p1 = (float(p0[0]) + 50.0 * float(gv[0]), float(p0[1]) + 50.0 * float(gv[1]))
                    draw_line(img, p0, p1, self.color_map["gaze"], t * 2)

            # ---- top-right text: hor / ver ----
            if hor_arr is not None and ver_arr is not None:
                hor = float(hor_arr[ix]) if np.isfinite(hor_arr[ix]) else np.nan
                ver = float(ver_arr[ix]) if np.isfinite(ver_arr[ix]) else np.nan

                # choose format (here: degrees)
                text = f"hor={hor:.2f}  ver={ver:.2f}" if np.isfinite(hor) and np.isfinite(ver) else "hor=nan  ver=nan"

                (tw, th), base = cv2.getTextSize(text, font, font_scale, thick_txt)
                x0 = max(0, W - tw - 2 * pad)
                y0 = max(0, 0 + pad + th)  # top-right

                # background box
                cv2.rectangle(
                    img,
                    (x0, y0 - th - pad),
                    (min(W - 1, x0 + tw + 2 * pad), min(H - 1, y0 + base + pad)),
                    (0, 0, 0),
                    thickness=-1,
                )
                # text
                cv2.putText(
                    img,
                    text,
                    (x0 + pad, y0),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thick_txt,
                    lineType=cv2.LINE_AA,
                )

            out[ix] = img
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