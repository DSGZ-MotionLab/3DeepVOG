import time, queue, threading, json
from pathlib import Path
from collections import deque
from fractions import Fraction

import numpy as np
import pandas as pd
import cv2
import torch
import skvideo.io as skv

class ResultCollector(threading.Thread):
    """
    Drains output queues and periodically writes (overwrites) result snapshots:
      - ellipses.pkl, gaze.pkl, torsion.pkl
    Optionally:
      - saves seg chunks (.npy)
      - writes seg overlay video (seg_overlay.mp4)
      - writes fitted overlay video (fitted_overlay.mp4)
    """

    def __init__(self, threads, args, daemon=True):
        super().__init__(daemon=daemon)
        self.threads = threads
        self.args = args

        pred_vid = args.get("pred_vid")
        if not pred_vid:
            raise ValueError("ResultCollector: args['pred_vid'] missing.")

        self.out_dir = Path(args["save_folder"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # outputs (overwrite snapshots)
        self.ellipse_pkl = self.out_dir / "ellipses.pkl"
        self.gaze_pkl = self.out_dir / "gaze.pkl"
        self.torsion_pkl = self.out_dir / "torsion.pkl"

        # seg chunks
        self.save_seg_chunks = bool(args.get("save_seg_chunks", False))
        self.segment_dir = self.out_dir / "segment_chunks"
        if self.save_seg_chunks:
            self.segment_dir.mkdir(parents=True, exist_ok=True)
        self.segment_chunk_idx = 0

        # video options
        self.write_seg_video = bool(args.get("write_seg_video", False))
        self.write_fit_video = bool(args.get("write_fit_video", False))
        self.alpha = float(args.get("seg_overlay_alpha", 0.35))

        self._fps = float(args.get("vid_fps", 30))
        self.W, self.H = int(args["vid_w"]), int(args["vid_h"])
        self._expected_rgb_shape = (self.H, self.W, 3)

        self.seg_video_path = self.out_dir / "seg_overlay.mp4"
        self.fit_video_path = self.out_dir / "fitted_overlay.mp4"
        self._writer = None
        self._fit_writer = None

        # buffers (overwrite snapshot = buffer content)
        self.ellipse_buf = []
        self.gaze_buf = []
        self.torsion_buf = []

        # pairing FIFOs
        self._frame_fifo = deque()
        self._seg_fifo = deque()

        # loop control
        self.flush_every = int(args.get("flush_every", 200))
        self._timeout = float(args.get("collector_timeout_sec", 0.01))
        self._idle_sleep = float(args.get("collector_idle_sleep_sec", 0.001))
        self._stop_requested = False
        self._NO_ITEM = object()

    # ---------- small utilities ----------
    def stop(self):
        self._stop_requested = True

    def _try_get(self, q):
        try:
            return q.get(timeout=self._timeout)
        except queue.Empty:
            return self._NO_ITEM

    @staticmethod
    def _atomic_pickle(obj, path: Path):
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        pd.to_pickle(obj, tmp)
        tmp.replace(path)

    def _open_ffmpeg_writer(self, path: Path):
        fps = float(self._fps)
        fps_frac = Fraction(fps).limit_denominator(1000)
        fps_str = f"{fps_frac.numerator}/{fps_frac.denominator}"
        return skv.FFmpegWriter(
            str(path),
            inputdict={"-r": fps_str},
            outputdict={
                "-r": fps_str,
                "-pix_fmt": "yuv420p",
                "-vcodec": "libx264",
                "-crf": "18",
                "-preset": "veryfast",
                "-movflags": "+faststart",
            },
        )

    def _ensure_bhwc3(self, seg):
        seg = seg.detach().cpu().numpy() if isinstance(seg, torch.Tensor) else np.asarray(seg)
        if seg.ndim == 3:
            seg = seg[..., None]
        if seg.ndim != 4:
            raise ValueError(f"seg expected 3/4 dims, got {seg.shape}")
        # (B,W,H,C) -> (B,H,W,C)
        if seg.shape[1] == self.W and seg.shape[2] == self.H:
            seg = np.transpose(seg, (0, 2, 1, 3))
        # (B,H,W,1) -> (B,H,W,3)
        if seg.shape[-1] == 1:
            seg = np.repeat(seg, 3, axis=-1)
        if seg.shape[-1] != 3:
            raise ValueError(f"seg channel must be 1 or 3, got {seg.shape}")
        return seg

    def _seg_to_overlay_bgr(self, seg_bhwc):
        # if grayscale replicated -> magenta mask
        is_gray = np.all(seg_bhwc[..., 0] == seg_bhwc[..., 1]) and np.all(seg_bhwc[..., 1] == seg_bhwc[..., 2])
        if is_gray:
            x = seg_bhwc[..., 0]
            if x.dtype == np.bool_:
                w = x.astype(np.uint8) * 255
            elif np.issubdtype(x.dtype, np.integer):
                w = (x > 0).astype(np.uint8) * 255
            else:
                w = (np.clip(x, 0, 1) * 255).astype(np.uint8)
            overlay = np.zeros((*w.shape, 3), dtype=np.uint8)
            overlay[..., 0] = w  # B
            overlay[..., 2] = w  # R
            return overlay

        # multi-class: 3 channels -> RGB masks -> BGR overlay
        colors = np.array([[0, 0, 255], [0, 255, 0], [255, 0, 0]], dtype=np.uint8)  # BGR
        if seg_bhwc.dtype == np.bool_:
            w = seg_bhwc.astype(np.uint8) * 255
        elif np.issubdtype(seg_bhwc.dtype, np.integer):
            w = (seg_bhwc > 0).astype(np.uint8) * 255
        else:
            w = (np.clip(seg_bhwc, 0, 1) * 255).astype(np.uint8)

        out = np.zeros((*w.shape[:-1], 3), dtype=np.uint8)
        for c in range(3):
            out = np.maximum(out, (w[..., c:c+1] * colors[c]).astype(np.uint8))
        return out

    def _write_seg_overlay_batch(self, frames_bgr, seg):
        if not self.write_seg_video:
            return
        if self._writer is None:
            self._writer = self._open_ffmpeg_writer(self.seg_video_path)

        seg = self._ensure_bhwc3(seg)

        # resize to (B,H,W,3)
        if (seg.shape[2], seg.shape[1]) != (self.W, self.H):
            seg2 = np.empty((seg.shape[0], self.H, self.W, 3), dtype=seg.dtype)
            for b in range(seg.shape[0]):
                for c in range(3):
                    seg2[b, :, :, c] = cv2.resize(seg[b, :, :, c], (self.W, self.H), interpolation=cv2.INTER_NEAREST)
            seg = seg2

        if (frames_bgr.shape[2], frames_bgr.shape[1]) != (self.W, self.H):
            frames_bgr = np.stack(
                [cv2.resize(frames_bgr[b], (self.W, self.H), interpolation=cv2.INTER_AREA) for b in range(frames_bgr.shape[0])],
                axis=0,
            )

        overlay = self._seg_to_overlay_bgr(seg)
        B = min(len(frames_bgr), len(overlay))
        for i in range(B):
            out = cv2.addWeighted(overlay[i], self.alpha, frames_bgr[i], 1.0 - self.alpha, 0)
            out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            self._writer.writeFrame(out_rgb)

    def _write_fit_frame(self, frame):
        if not self.write_fit_video:
            return
        if self._fit_writer is None:
            self._fit_writer = self._open_ffmpeg_writer(self.fit_video_path)

        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()
        frame = np.asarray(frame)

        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_BGR2RGB)

        if rgb.shape != self._expected_rgb_shape:
            rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)
            if rgb.ndim == 2:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        self._fit_writer.writeFrame(rgb)

    def _flush_all(self):
        if self.ellipse_buf:
            df = pd.concat(self.ellipse_buf, ignore_index=True)
            self._atomic_pickle(df, self.ellipse_pkl)
            self.ellipse_buf.clear()

        if self.gaze_buf:
            df = pd.concat(self.gaze_buf, ignore_index=True)
            self._atomic_pickle(df, self.gaze_pkl)
            self.gaze_buf.clear()

        if self.torsion_buf:
            self._atomic_pickle(list(self.torsion_buf), self.torsion_pkl)
            self.torsion_buf.clear()

    # ---------- main loop ----------
    def run(self):
        ques = self.threads["ques"]

        done = {
            "frame_out": False,
            "ellipse_out": False,
            "segment_out": (ques.get("segment_out") is None),
            "gaze_out": (ques.get("gaze_out") is None),
            "torsion_out": (ques.get("torsion_out") is None),
            "fitted_frame_out": (ques.get("fitted_frame_out") is None),
        }

        ticks = 0
        while not all(done.values()) and not self._stop_requested:
            progressed = False

            # frames (for seg overlay pairing)
            x = self._try_get(ques["frame_out"])
            if x is not self._NO_ITEM:
                progressed = True
                done["frame_out"] = (x is None)
                if x is not None:
                    self._frame_fifo.append(x)

            # ellipse
            if not done["ellipse_out"]:
                x = self._try_get(ques["ellipse_out"])
                if x is not self._NO_ITEM:
                    progressed = True
                    done["ellipse_out"] = (x is None)
                    if x is not None:
                        df = x if isinstance(x, pd.DataFrame) else pd.DataFrame(x if isinstance(x, list) else [x])
                        self.ellipse_buf.append(df)

            # segment
            if not done["segment_out"]:
                x = self._try_get(ques["segment_out"])
                if x is not self._NO_ITEM:
                    progressed = True
                    done["segment_out"] = (x is None)
                    if x is not None:
                        seg_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
                        if self.save_seg_chunks:
                            np.save(self.segment_dir / f"seg_{self.segment_chunk_idx:06d}.npy", seg_np)
                            self.segment_chunk_idx += 1
                        self._seg_fifo.append(seg_np)

            # pair + write overlay
            while self._frame_fifo and self._seg_fifo:
                self._write_seg_overlay_batch(self._frame_fifo.popleft(), self._seg_fifo.popleft())

            # fitted frames
            if not done["fitted_frame_out"]:
                x = self._try_get(ques["fitted_frame_out"])
                if x is not self._NO_ITEM:
                    progressed = True
                    done["fitted_frame_out"] = (x is None)
                    if x is not None:
                        fit_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
                        if fit_np.ndim == 4:
                            for i in range(fit_np.shape[0]):
                                self._write_fit_frame(fit_np[i])
                        else:
                            self._write_fit_frame(fit_np)

            # gaze
            if not done["gaze_out"]:
                x = self._try_get(ques["gaze_out"])
                if x is not self._NO_ITEM:
                    progressed = True
                    done["gaze_out"] = (x is None)
                    if x is not None:
                        self.gaze_buf.append(pd.DataFrame(x))

            # torsion
            if not done["torsion_out"]:
                x = self._try_get(ques["torsion_out"])
                if x is not self._NO_ITEM:
                    progressed = True
                    done["torsion_out"] = (x is None)
                    if x is not None:
                        angles = x.get("torsion_angles") if isinstance(x, dict) else x
                        if isinstance(angles, torch.Tensor):
                            angles = angles.detach().cpu().numpy()
                        if isinstance(angles, np.ndarray):
                            angles = angles.tolist()
                        if not isinstance(angles, list):
                            angles = [float(angles)]
                        self.torsion_buf.extend(angles)

            ticks += 1
            if ticks % self.flush_every == 0:
                self._flush_all()

            if not progressed:
                time.sleep(self._idle_sleep)

        self._flush_all()
        if self._writer is not None:
            self._writer.close()
        if self._fit_writer is not None:
            self._fit_writer.close()