import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import threading
# from itertools import product
# import skvideo.io as skv
# from deepvog3D.draw_ellipse_batch import fit_ellipse_compact
from ..models.deepvog3d_model import Model_3DeepVOG

class ModelInference(threading.Thread):
    def __init__(self, threads, args, daemon=False, use_queue=True):
        super().__init__(daemon=daemon)
        self.name = 'Thread-ModelInference'
        self.threads = threads
        self.params = args
        self.use_queue = args.get('is_parallel', use_queue)  #priority to args['is_parallel'] if it exists
        self.model = Model_3DeepVOG(video_width=args['vid_w'], 
                                    video_height=args['vid_h'],
                                    device=args['device'], 
                                    ff_model_weights= args['segmentation_model_weights_path'],
                                    model=args['segmentation_model'])
        self.params['segmentation_model'] = self.model.model    #if this param is not defined in the main
        self.elapsed_time = 0

    def run(self):
        while True:
            # read data from source queue
            frame_batch = self.threads['ques']['model_inference'].get()
            if frame_batch is None: 
                # received poison pill, pass on!    Never goes this place!!! Never pass on!!
                self.threads['ques']['post_processing'].put(None)
                break
            else:
                self.model_inference(frame_batch)
    
    def model_inference(self, frame_batch):
        time00 = time.time()
        preds_batch = self.model.predict(frame_batch['imgs'])
        frame_batch['segs'] = preds_batch
        time01 = time.time()
        self.elapsed_time += (time01 - time00)
        # pass on to post_processing
        #debugging
        # plt.imshow(preds_batch[9, :, :, 1].cpu().numpy())
        # plt.savefig("debug_output.png")  # Save to file instead of showing
        if self.use_queue:
            self.threads['ques']['post_processing'].put(frame_batch)
        else:
            return frame_batch

        