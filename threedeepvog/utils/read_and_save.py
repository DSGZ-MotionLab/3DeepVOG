import pickle
import torch
import numpy as np
import cv2
import os
from pathlib import Path
import json

#original training image size is 240x320
#therefore, the scaling factor is calculated as follows:
#image_scaling_factor = np.linalg.norm((240, 320)) / np.linalg.norm((h, w))

# Path to the video file
def get_video_info(video_src):
    video_name_with_ext = os.path.split(video_src)[1]
    video_name_root, ext = os.path.splitext(video_name_with_ext)
    vreader = cv2.VideoCapture(video_src)
    m = int(vreader.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(vreader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(vreader.get(cv2.CAP_PROP_FRAME_WIDTH))
    if vreader.get(cv2.CAP_PROP_MONOCHROME):
        channels = 1
    else:
        channels = 3
    fps = vreader.get(cv2.CAP_PROP_FPS)
    image_scaling_factor = np.linalg.norm((240, 320)) / np.linalg.norm((h, w))
    shape_correct = (h,w)==(240,320)
    return video_name_root, ext, vreader, (m, h, w, channels), shape_correct, image_scaling_factor, fps

def save_pkl(filename, params):
    with open(filename, 'wb') as f:
        pickle.dump(params, f, protocol=pickle.HIGHEST_PROTOCOL)


def get_video_info_torch(video_src):
    video_name_with_ext = os.path.split(video_src)[1]
    video_name_root, ext = os.path.splitext(video_name_with_ext)
    vreader = cv2.VideoCapture(video_src)
    m = int(vreader.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(vreader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(vreader.get(cv2.CAP_PROP_FRAME_WIDTH))
    channels = 1 if vreader.get(cv2.CAP_PROP_MONOCHROME) else 3
    fps = vreader.get(cv2.CAP_PROP_FPS)
    image_scaling_factor = torch.linalg.norm(torch.tensor((240, 320), dtype=torch.float32)) / torch.linalg.norm(torch.tensor((h, w), dtype=torch.float32))
    shape_correct = (h, w) == (240, 320)
    return video_name_root, ext, vreader, (m, h, w, channels), shape_correct, image_scaling_factor, fps


def get_video_info_cv2(video_path):
    vid_cap = cv2.VideoCapture(video_path)
    width = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_cap.release()
    return width, height, fps, total_frames


def save_json(path: str, save_dict: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)  # <-- create folders
    json_str = json.dumps(save_dict, indent=4)
    path.write_text(json_str, encoding="utf-8")
        
def load_json(path):
    with open(path, "r+") as fh:
        json_str = fh.read()
    return json.loads(json_str)

def csv_reader(csv_path):
    col_dict = dict()
    col_list = []
    with open(csv_path, "r") as fh:
        for idx, line in enumerate(fh):
            row = line.split(",")
            row_stripped = list(map(lambda x : x.strip(), row))
            if idx == 0:
                for col in row_stripped:
                    col_list.append(col)
                    col_dict[str(col)] = []
            else:
                for col_idx, col in enumerate(row_stripped):
                    col_dict[col_list[col_idx]].append(str(col))
    return col_dict


if __name__ == "__main__":
    pass
