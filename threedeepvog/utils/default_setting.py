# default parameters
def default_params():
    params = dict()
    params['batch_size'] = 32
    params['threshold_pupil'] = 0.5
    params['threshold_iris'] = 0.5
    params['threshold_glints'] = 0.5
    params['threshold_eyeinner'] = 0.5
    params['threshold_confidence_pupil'] = 0.96
    params['viz_frame_interval'] = 1
    params['alpha_seg_overlay'] = 0.3
    params['viz_time_range'] = 5.0
    params['th_under_exposure'] = 0.1  #if too low, corrupted frame will potentailly cause system collapse
    params['th_over_exposure'] = 0.9
    return params