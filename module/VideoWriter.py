import numpy as np
import threading
import skvideo.io as skv
import cv2
import subprocess

def round_up_to_odd_int(f):
    return int(np.ceil(f) // 2 * 2 + 1)

def visualization_merger(params, method='ffmpeg'):
    '''
    - read queues for viz_seg, viz_pupil_params, viz_ellipse, viz_gaze, viz_torsion
    - vstack the images
    - send frame_batch to VideoWriter.writeFrame
    '''
    vid_files = []
    if params['viz_segmentation']:
        vid_files.append(params['viz_filename_mp4_ellipses'])
    if params['viz_gaze']:
        pass
    if params['viz_torsion']:
        vid_files.append(params['viz_filename_mp4_torsion'])
    
    if len(vid_files) > 1:
        if method == 'ffmpeg':
            cmd_ffmpeg = [
                'ffmpeg'
            ]
            for filepath in vid_files:
                cmd_ffmpeg.extend(['-i', filepath])
            cmd_ffmpeg.extend([
                '-filter_complex', f'vstack=inputs={len(vid_files)}',
                params['viz_filename_mp4']
            ])
            print('visualization_merger ffmpeg cmd:')
            print(' '.join(cmd_ffmpeg))
            ret_val = subprocess.run(cmd_ffmpeg)
        else:
            ret_val = -1
    else: 
        ret_val = -1
    return ret_val

class VideoWriter(threading.Thread):
    def __init__(self, threads, args, use_ffmpeg=True, filename_override=None, src_que='video_writer_ellipses', daemon=False, use_queue=True):
        super().__init__(daemon=daemon)
        self.name = 'Thread-' + src_que
        self.threads = threads
        self.params = args
        self.use_queue = args.get('is_parallel', use_queue)  #priority to args['is_parallel'] if it exists
        self.elapsed_time = 0
        self.fps = args['vid_fps']
        self.viz_frame_interval = args['viz_frame_interval']
        self.src_que = src_que
        self.use_ffmpeg = use_ffmpeg

        if args['viz_frame_interval'] > 1:
            self.fps = self.fps / args['viz_frame_interval']

        self.viz_filename_mp4 = args['viz_filename_mp4']
        if filename_override is not None:
            self.viz_filename_mp4 = filename_override

        if self.use_ffmpeg:
            self.inputdict = {'-r': str(self.fps)}
            self.outputdict = {
                '-vcodec': 'libx264',
                '-pix_fmt': 'yuv420p',
                '-r': str(self.fps),
                '-crf': '15'
            }
            self.vwriter = skv.FFmpegWriter(
                self.viz_filename_mp4,
                inputdict=self.inputdict,
                outputdict=self.outputdict
            )
        else:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.vwriter = cv2.VideoWriter(self.viz_filename_mp4, fourcc, self.fps, (args['vid_w'], args['vid_h']))

    def run(self):
        while True:
            frame_batch = self.threads['ques'][self.src_que].get()
            if frame_batch is None: 
                print('Videowriter "%s": received poison pill! Closing vwriter.' % self.name)
                self.release()
                break
            else:
                if self.viz_frame_interval == 1:
                    self.write_frame_batch(frame_batch)
                else:
                    self.write_single_frame(frame_batch)
    
    def write_single_frame(self, viz_frame):
        if self.use_ffmpeg:
            self.vwriter.writeFrame(viz_frame['img'])
        else:
            self.vwriter.write(viz_frame['img'])

        if viz_frame['idx'] % 100 == 0:
            pass
            # print(f'Wrote frame {viz_frame["idx"]} to vid ({self.viz_filename_mp4})')

    def write_frame_batch(self, viz_frames):
        for idx, viz_frame in enumerate(viz_frames):
            if self.use_ffmpeg:
                self.vwriter.writeFrame(viz_frame['img'])
            else:
                self.vwriter.write(viz_frame['img'])

            if viz_frame['idx'] % 100 == 0:
                pass
                # print(f'Wrote frame {viz_frame["idx"]} to vid ({self.viz_filename_mp4})')

    def release(self):
        print('Releasing video writer...')
        if self.use_ffmpeg:
            self.vwriter.close()
        else:
            self.vwriter.release()