from collections import defaultdict
import threading, queue, time
from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd
import cv2
import skvideo.io as skv
import torch
import glob, os

_STOP = object()
class DiskWriter(threading.Thread):
    """
    Background writer thread:
    - Receives batched outputs via a queue.
    - Buffers them in RAM for a while (flush_every).
    - Periodically writes atomic pickle files to disk.

    Why buffering?
    - Writing every batch is slow (I/O bound) and can bottleneck the pipeline.
    - Buffering + periodic flush reduces overhead and keeps processing realtime.
    """
    def __init__(self, out_dir: Path, flush_every=200, daemon=True):
        super().__init__(daemon=daemon)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.flush_every = int(flush_every)
        self.q = queue.Queue(maxsize=64)

        self.ellipse_buf = []
        self.gaze_buf = defaultdict(list)
        self.torsion_buf = []

        # chunk counters
        self.ellipse_i = 0
        self.gaze_i = 0
        self.torsion_i = 0
        self.ticks = 0

    @staticmethod
    def _atomic_pickle(obj, path: Path):
        tmp = path.with_suffix(path.suffix + ".tmp")
        pd.to_pickle(obj, tmp)
        tmp.replace(path)

    def push(self, kind, payload):
        self.q.put((kind, payload))

    def push_gaze(self, payload: dict):
        for k, v in payload.items():
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            self.gaze_buf[k].append(v)

    def _flush(self):
        # ---- ellipses chunk ----
        if self.ellipse_buf:
            df = pd.concat(self.ellipse_buf, ignore_index=True)
            self.ellipse_i += 1
            self._atomic_pickle(df, self.out_dir / f"ellipses_{self.ellipse_i:06d}.pkl")
            self.ellipse_buf.clear()

        # ---- gaze chunk ----
        if self.gaze_buf:
            big = {k: np.concatenate(v, axis=0) for k, v in self.gaze_buf.items()}

            vec_names = {
                "c_eye": ("x", "y", "z"),
                "c_pupil": ("x", "y", "z"),
                "gaze": ("x", "y", "z"),
                "c_eye2d": ("x", "y"),
                "location": ("x", "y"),
                "norm_pos": ("x", "y"),
                "entpup_el": ("rad", "cx", "cy", "a", "b"),
            }

            cols = {}
            for k, a in big.items():
                a = np.asarray(a)
                if a.ndim == 1:
                    cols[k] = a
                elif a.ndim == 2:
                    names = vec_names.get(k)
                    for j in range(a.shape[1]):
                        suffix = names[j] if names and j < len(names) else str(j)
                        cols[f"{k}_{suffix}"] = a[:, j]
                else:
                    cols[k] = list(a)

            self.gaze_i += 1
            self._atomic_pickle(pd.DataFrame(cols), self.out_dir / f"gaze_{self.gaze_i:06d}.pkl")
            self.gaze_buf.clear()

        # ---- torsion chunk ----
        if self.torsion_buf:
            self.torsion_i += 1
            self._atomic_pickle(list(self.torsion_buf), self.out_dir / f"torsion_{self.torsion_i:06d}.pkl")
            self.torsion_buf.clear()

    def run(self):
        while True:
            item = self.q.get()
            if item is _STOP:
                break

            kind, payload = item

            if kind == "ellipse":
                df = payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload if isinstance(payload, list) else [payload])
                self.ellipse_buf.append(df)

            elif kind == "gaze":
                self.push_gaze(payload)

            elif kind == "torsion":
                angles = payload.get("torsion_angles") if isinstance(payload, dict) else payload
                if isinstance(angles, torch.Tensor):
                    angles = angles.detach().cpu().numpy()
                if isinstance(angles, np.ndarray):
                    angles = angles.tolist()
                if not isinstance(angles, list):
                    angles = [float(angles)]
                self.torsion_buf.extend(angles)

            self.ticks += 1
            if self.ticks % self.flush_every == 0:
                self._flush()

        self._flush()


class OverlayWriter(threading.Thread):
    """
    Writes a segmentation overlay video asynchronously.
    It pairs raw frames with segmentation masks using a shared batch_id.
    """
    def __init__(self, out_path: Path, fps: float, size_wh, alpha=0.35, daemon=True):
        super().__init__(daemon=daemon)
        self.out_path = Path(out_path)
        self.fps = float(fps)
        self.W, self.H = map(int, size_wh)
        self.alpha = float(alpha)
        self.q = queue.Queue(maxsize=16)
        self.writer = None

        # pairing by batch_id
        self.frames = {}
        self.segs = {}

    def _open(self):
        if self.writer is not None:
            return
        fps_frac = Fraction(self.fps).limit_denominator(1000)
        fps_str = f"{fps_frac.numerator}/{fps_frac.denominator}"
        self.writer = skv.FFmpegWriter(
            str(self.out_path),
            inputdict={"-r": fps_str},
            outputdict={"-r": fps_str, "-pix_fmt": "yuv420p", "-vcodec": "libx264",
                        "-crf": "18", "-preset": "veryfast", "-movflags": "+faststart"}
        )

    def push_frames(self, batch_id, frames_bgr):
        self.q.put(("frames", batch_id, frames_bgr))

    def push_seg(self, batch_id, seg):
        self.q.put(("seg", batch_id, seg))

    def _ensure_bhwc3(self, seg):
        """
        Normalize segmentation format to (B,H,W,3).
        Handles tensors, grayscale masks, and (B,W,H,C) layout.
        """
        seg = seg.detach().cpu().numpy() if isinstance(seg, torch.Tensor) else np.asarray(seg)
        if seg.ndim == 3: seg = seg[..., None]
        if seg.shape[-1] == 1: seg = np.repeat(seg, 3, axis=-1)
        # (B,W,H,C) -> (B,H,W,C)
        if seg.shape[1] == self.W and seg.shape[2] == self.H:
            seg = np.transpose(seg, (0,2,1,3))
        return seg

    def _seg_to_overlay_bgr(self, seg_bhwc):
        """
        Convert seg array to a visible overlay in BGR.
        If grayscale mask -> magenta overlay.
        """
        # simple magenta if grayscale mask
        is_gray = np.all(seg_bhwc[...,0] == seg_bhwc[...,1]) and np.all(seg_bhwc[...,1] == seg_bhwc[...,2])
        if is_gray:
            x = seg_bhwc[...,0]
            w = (x > 0).astype(np.uint8) * 255 if np.issubdtype(x.dtype, np.integer) else (np.clip(x,0,1)*255).astype(np.uint8)
            ov = np.zeros((*w.shape,3), np.uint8)
            ov[...,0] = w; ov[...,2] = w
            return ov
        # otherwise assume already RGB-ish masks, just threshold into magenta-ish overlay
        w = (np.clip(seg_bhwc,0,1)*255).astype(np.uint8) if not np.issubdtype(seg_bhwc.dtype, np.integer) else (seg_bhwc>0).astype(np.uint8)*255
        ov = np.zeros((*w.shape[:-1],3), np.uint8)
        ov[...,2] = w[...,0]  # R
        ov[...,1] = w[...,1]  # G
        ov[...,0] = w[...,2]  # B
        return ov

    def _write_pair(self, frames_bgr, seg):
        self._open()
        seg = self._ensure_bhwc3(seg)

        # resize if needed
        if (frames_bgr.shape[2], frames_bgr.shape[1]) != (self.W, self.H):
            frames_bgr = np.stack([cv2.resize(frames_bgr[b], (self.W,self.H)) for b in range(frames_bgr.shape[0])], axis=0)
        if (seg.shape[2], seg.shape[1]) != (self.W, self.H):
            seg2 = np.empty((seg.shape[0], self.H, self.W, 3), dtype=seg.dtype)
            for b in range(seg.shape[0]):
                for c in range(3):
                    seg2[b,:,:,c] = cv2.resize(seg[b,:,:,c], (self.W,self.H), interpolation=cv2.INTER_NEAREST)
            seg = seg2

        overlay = self._seg_to_overlay_bgr(seg)
        B = min(frames_bgr.shape[0], overlay.shape[0])
        for i in range(B):
            out = cv2.addWeighted(overlay[i], self.alpha, frames_bgr[i], 1.0-self.alpha, 0)
            self.writer.writeFrame(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

    def run(self):
        """
        Pairing logic:
        - Store frames/segs by batch id
        - When both exist for a batch id, write them and delete from dicts
        """
        while True:
            item = self.q.get()
            if item is _STOP:
                break
            kind, bid, payload = item
            if kind == "frames":
                self.frames[bid] = payload
            else:
                self.segs[bid] = payload

            if bid in self.frames and bid in self.segs:
                self._write_pair(self.frames.pop(bid), self.segs.pop(bid))

        if self.writer is not None:
            self.writer.close()



class FitVideoWriter(threading.Thread):
    """
    Writes fitted/overlay frames to video asynchronously.
    Input can be batched or single frame, grayscale or BGR.
    """
    def __init__(self, out_path: Path, fps: float, size_wh, daemon=True):
        super().__init__(daemon=daemon)
        self.out_path = Path(out_path)
        self.fps = float(fps)
        self.W, self.H = map(int, size_wh)
        self.q = queue.Queue(maxsize=16)
        self.writer = None

    def _open(self):
        if self.writer is not None:
            return
        fps_frac = Fraction(self.fps).limit_denominator(1000)
        fps_str = f"{fps_frac.numerator}/{fps_frac.denominator}"
        self.writer = skv.FFmpegWriter(
            str(self.out_path),
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

    def push(self, frames):
        self.q.put(frames)

    def run(self):
        while True:
            frames = self.q.get()
            if frames is _STOP:
                break

            self._open()

            if isinstance(frames, torch.Tensor):
                frames = frames.detach().cpu().numpy()
            frames = np.asarray(frames)

            # frames: (B,H,W,3) or (B,H,W) or (H,W,3) or (H,W)
            if frames.ndim == 2 or frames.ndim == 3:
                frames = frames[None, ...]  # -> (1, ...)

            for i in range(frames.shape[0]):
                f = frames[i]
                if f.ndim == 2:
                    rgb = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                else:
                    # assume BGR
                    rgb = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2RGB)

                if (rgb.shape[1], rgb.shape[0]) != (self.W, self.H):
                    rgb = cv2.resize(rgb, (self.W, self.H), interpolation=cv2.INTER_AREA)

                self.writer.writeFrame(rgb)

        if self.writer is not None:
            self.writer.close()



class ResultRouter(threading.Thread):
    """
    Lightweight collector/router thread:
    - Drains shared output queues (frame_out, segment_out, ellipse_out, gaze_out, torsion_out, fitted_frame_out)
    - Forwards to:
        * DiskWriter (pickle files)
        * OverlayWriter (seg overlay mp4)
        * FitVideoWriter (fit overlay mp4)
    This keeps heavy work away from compute threads.
    """
    def __init__(self, threads, args, disk, overlay=None, fit_writer=None, daemon=True):
        super().__init__(daemon=daemon)
        self.threads = threads
        self.args = args
        self.disk = disk
        self.overlay = overlay
        self.fit_writer = fit_writer
        self._timeout = float(args.get("collector_timeout_sec", 0.01))
        self._idle_sleep = float(args.get("collector_idle_sleep_sec", 0.001))
        self._NO_ITEM = object()

    def _try_get(self, q):
        try:
            return q.get(timeout=self._timeout)
        except queue.Empty:
            return self._NO_ITEM

    def run(self):
        q = self.threads["ques"]

        done = {
            "frame_out": False,
            "ellipse_out": False,
            "segment_out": (q.get("segment_out") is None),
            "gaze_out": (q.get("gaze_out") is None),
            "torsion_out": (q.get("torsion_out") is None),
            "fitted_frame_out": (q.get("fitted_frame_out") is None),
        }

        while not all(done.values()):
            progressed = False

            # frames for overlay: expect (bid, frames_bgr)
            x = self._try_get(q["frame_out"])
            if x is not self._NO_ITEM:
                progressed = True
                if x is None:
                    done["frame_out"] = True
                else:
                    if self.overlay:
                        bid, frames_bgr = x
                        self.overlay.push_frames(bid, frames_bgr)

            # segment for overlay: expect (bid, seg)
            if not done["segment_out"]:
                x = self._try_get(q["segment_out"])
                if x is not self._NO_ITEM:
                    progressed = True
                    if x is None:
                        done["segment_out"] = True
                    else:
                        if self.overlay:
                            bid, seg = x
                            self.overlay.push_seg(bid, seg)

                        # fitted frames -> fit writer
            if not done["fitted_frame_out"]:
                x = self._try_get(q["fitted_frame_out"])
                if x is not self._NO_ITEM:
                    progressed = True
                    if x is None:
                        done["fitted_frame_out"] = True
                    else:
                        if self.fit_writer:
                            self.fit_writer.push(x)

            # ellipse → disk
            if not done["ellipse_out"]:
                x = self._try_get(q["ellipse_out"])
                if x is not self._NO_ITEM:
                    progressed = True
                    if x is None:
                        done["ellipse_out"] = True
                    else:
                        self.disk.push("ellipse", x)

            # gaze → disk
            if not done["gaze_out"]:
                x = self._try_get(q["gaze_out"])
                if x is not self._NO_ITEM:
                    progressed = True
                    if x is None:
                        done["gaze_out"] = True
                    else:
                        self.disk.push("gaze", x)

            # torsion → disk
            if not done["torsion_out"]:
                x = self._try_get(q["torsion_out"])
                if x is not self._NO_ITEM:
                    progressed = True
                    if x is None:
                        done["torsion_out"] = True
                    else:
                        self.disk.push("torsion", x)

            if not progressed:
                time.sleep(self._idle_sleep)

        # stop workers
        self.disk.q.put(_STOP)
        if self.overlay:
            self.overlay.q.put(_STOP)
        if self.fit_writer:
            self.fit_writer.q.put(_STOP)



def merge_chunks(save_folder: str, cleanup: bool = True):
    logdir = str(save_folder)

    ell_files  = sorted(glob.glob(os.path.join(logdir, "ellipses_*.pkl")))
    gaze_files = sorted(glob.glob(os.path.join(logdir, "gaze_*.pkl")))
    tor_files  = sorted(glob.glob(os.path.join(logdir, "torsion_*.pkl")))

    # ---- merge ellipses ----
    if ell_files:
        ell = pd.concat([pd.read_pickle(p) for p in ell_files], ignore_index=True)
        ell.to_pickle(os.path.join(logdir, "ellipses.pkl"))

    # ---- merge gaze ----
    if gaze_files:
        gaze = pd.concat([pd.read_pickle(p) for p in gaze_files], ignore_index=True)
        gaze.to_pickle(os.path.join(logdir, "gaze.pkl"))

    # ---- merge torsion ----
    if tor_files:
        tors = []
        for p in tor_files:
            tors.extend(pd.read_pickle(p))
        pd.to_pickle(tors, os.path.join(logdir, "torsion.pkl"))

    # ---- cleanup temp chunk files ----
    if cleanup:
        for p in ell_files + gaze_files + tor_files:
            try:
                os.remove(p)
            except OSError as e:
                print(f"Warning: could not delete {p}: {e}")