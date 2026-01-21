import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from scipy.ndimage import label
from scipy.optimize import minimize

def gaze_extract_LG(df_LG_gaze_out, thr_confidence = 0.96):
    # gaze_x_LG = np.rad2deg(np.arctan(np.deg2rad(df_LG_gaze_out.gaze_x.values)))
    # gaze_y_LG = np.rad2deg(np.arctan(np.deg2rad(df_LG_gaze_out.gaze_y.values)))
    gaze_x_LG = df_LG_gaze_out.gaze_x.values
    gaze_y_LG = df_LG_gaze_out.gaze_y.values
    gaze_x_LG[df_LG_gaze_out.confidence<thr_confidence] = np.nan
    gaze_y_LG[df_LG_gaze_out.confidence<thr_confidence] = np.nan
    gaze_x_LG = gaze_x_LG - np.nanmean(gaze_x_LG)
    gaze_y_LG = gaze_y_LG - np.nanmean(gaze_y_LG)
    return gaze_x_LG, gaze_y_LG

def gaze_extract_GT(df_GT_gaze_out):
    # gaze_y_GT = np.degrees(2*np.arctan(df_GT_gaze_out.ver.values))
    # gaze_x_GT = np.degrees(2*np.arctan(df_GT_gaze_out.hor.values)) 
    gaze_y_GT = df_GT_gaze_out.LeftEyeVer_Deg.values
    gaze_x_GT = df_GT_gaze_out.LeftEyeHor_Deg.values
    gaze_y_GT = gaze_y_GT - np.nanmean(gaze_y_GT)
    gaze_x_GT = gaze_x_GT - np.nanmean(gaze_x_GT)
    return gaze_x_GT, gaze_y_GT

def gaze_extract_PL(df, thr_confidence = 0.96):   #Also apply for deepPL
    gaze_y_PL, gaze_x_PL = df.theta.values - np.pi/2, df.phi.values + np.pi/2
    PL_confidence = df.confidence.values
    PL_model_confidence = df.model_confidence.values
    gaze_y_PL[PL_confidence<thr_confidence] = np.nan
    gaze_x_PL[PL_confidence<thr_confidence] = np.nan
    gaze_y_PL[PL_model_confidence<1] = np.nan
    gaze_x_PL[PL_model_confidence<1] = np.nan
    gaze_x_PL, gaze_y_PL = np.rad2deg(gaze_x_PL), np.rad2deg(gaze_y_PL)
    gaze_x_PL, gaze_y_PL = gaze_x_PL - np.nanmean(gaze_x_PL), gaze_y_PL - np.nanmean(gaze_y_PL)
    return -gaze_x_PL, gaze_y_PL


def read_calib_GT(df_calib_GT):
    calib_x_GT = np.degrees(2*np.arctan(df_calib_GT.LeftEyeHor_Rot.values)) 
    calib_y_GT = np.degrees(2*np.arctan(df_calib_GT.LeftEyeVer_Rot.values))
    calib_x_GT = calib_x_GT - np.nanmean(calib_x_GT)
    calib_y_GT = calib_y_GT - np.nanmean(calib_y_GT)
    return calib_x_GT, -calib_y_GT

def gaze_extract_GT_v2(df_GT):
    calib_x_GT = np.degrees(2*np.arctan(df_GT.LeftEyeHor_Rot.values)) 
    calib_y_GT = np.degrees(2*np.arctan(df_GT.LeftEyeVer_Rot.values))
    calib_x_GT = calib_x_GT - np.nanmean(calib_x_GT)
    calib_y_GT = calib_y_GT - np.nanmean(calib_y_GT)
    return calib_x_GT, -calib_y_GT

def cart2sph_batch_PL(N, eps=1e-8):
    N /= np.where((norms := np.linalg.norm(N, axis=0, keepdims=True)) < eps, 1.0, norms)
    phi = np.arctan2(N[2], N[0]) + np.pi / 2
    theta = np.arccos(N[1]) - np.pi / 2
    return -np.rad2deg(phi), np.rad2deg(theta)


def opt_transform_v2(df_GT, normals, conf, thr_confidence=0.96, default_M=np.eye(3), opt_flag=True):
    GT_x, GT_y = np.rad2deg(df_GT.hor), -np.rad2deg(df_GT.ver)
    GT_x -= np.nanmean(GT_x); GT_y -= np.nanmean(GT_y)
    pred_full = np.full_like(GT_x, np.nan); pred_y_full = np.full_like(GT_y, np.nan)
    valid = ~np.isnan(GT_x) & ~np.isnan(GT_y) & ~np.isnan(normals).any(0)
    GT_x, GT_y, normals, conf = GT_x[valid], GT_y[valid], normals[:, valid], conf[valid]
    def loss(params):
        M = params.reshape(3, 3)
        pred_x, pred_y = cart2sph_batch_PL(M @ normals)
        pred_x[conf < thr_confidence] = np.nan; pred_y[conf < thr_confidence] = np.nan
        pred_x -= np.nanmean(pred_x); pred_y -= np.nanmean(pred_y)
        return np.nanmedian(np.abs(pred_x - GT_x)) + np.nanmedian(np.abs(pred_y - GT_y))
    M_opt = minimize(loss, default_M.flatten(), method='L-BFGS-B').x.reshape(3, 3) if opt_flag else default_M
    pred = M_opt @ normals
    fx, fy = cart2sph_batch_PL(pred)
    fx[conf < thr_confidence] = np.nan; fy[conf < thr_confidence] = np.nan
    fx -= np.nanmean(fx); fy -= np.nanmean(fy)
    pred_full[valid], pred_y_full[valid] = fx, fy
    return pred_full, pred_y_full, M_opt


def opt_transform_v3(GT_list, normals, conf, thr_confidence=0.96, default_M=np.eye(3), opt_flag=True):
    GT_x, GT_y = GT_list[0], GT_list[1]
    pred_full = np.full_like(GT_x, np.nan); pred_y_full = np.full_like(GT_y, np.nan)
    valid = ~np.isnan(GT_x) & ~np.isnan(GT_y) & ~np.isnan(normals).any(0)
    GT_x, GT_y, normals, conf = GT_x[valid], GT_y[valid], normals[:, valid], conf[valid]
    def loss(params):
        M = params.reshape(3, 3)
        pred_x, pred_y = cart2sph_batch_PL(M @ normals)
        pred_x[conf < thr_confidence] = np.nan; pred_y[conf < thr_confidence] = np.nan
        pred_x -= np.nanmean(pred_x); pred_y -= np.nanmean(pred_y)
        return np.nanmedian(np.abs(pred_x - GT_x)) + np.nanmedian(np.abs(pred_y - GT_y))
    M_opt = minimize(loss, default_M.flatten(), method='L-BFGS-B').x.reshape(3, 3) if opt_flag else default_M
    pred = M_opt @ normals
    fx, fy = cart2sph_batch_PL(pred)
    fx[conf < thr_confidence] = np.nan; fy[conf < thr_confidence] = np.nan
    fx -= np.nanmean(fx); fy -= np.nanmean(fy)
    pred_full[valid], pred_y_full[valid] = fx, fy
    return pred_full, pred_y_full, M_opt


def estimate_calib_matrix(df_gaze_out, thr_confidence = 0.975):  
    gaze_x_PL, gaze_y_PL = gaze_extract_PL(df_gaze_out, thr_confidence = 0.96)
    pupil_c_x = []
    pupil_c_y = []
    for x,y in df_gaze_out.norm_pos.values:
        pupil_c_x.append(x)
        pupil_c_y.append(y)
    model_confidence = df_gaze_out.model_confidence.values
    X = np.stack([np.array(pupil_c_x), np.array(pupil_c_y), np.ones_like(pupil_c_x)], axis=0)
    Y = np.stack([gaze_x_PL, gaze_y_PL, np.ones_like(gaze_x_PL)], axis=0)
    X_interest = X[:,model_confidence==1]
    Y_interest = Y[:,model_confidence==1]
    x_pinv = np.linalg.pinv(X_interest)
    M = np.dot(Y_interest, x_pinv)
    Y_pred = np.dot(M, X)
    return Y_pred

def gaze_extract_norm(df_PL_gaze_out, thr_confidence = 0.90):   #Also apply for deepPL
    norm_pos = df_PL_gaze_out.norm_pos.values
    norm_x = []
    norm_y = []
    for x,y in norm_pos:
        norm_x.append(x)
        norm_y.append(y)
    norm_x = np.array(norm_x)
    norm_y = np.array(norm_y)
    PL_model_confidence = df_PL_gaze_out.model_confidence.values
    # norm_x[PL_model_confidence==0] = np.nan
    # norm_y[PL_model_confidence==0] = np.nan
    norm_x = norm_x - np.nanmean(norm_x)
    norm_y = norm_y - np.nanmean(norm_y)
    return -100*norm_x, 80*norm_y

def cut_unstable_period(arr: np.ndarray, delta: int = 100) -> np.ndarray:
    """
    Removes unstable periods from a boolean array by setting the first `delta` 
    and last `delta` True values of each cluster to False.

    Parameters:
    - arr: np.ndarray
        Boolean array representing the input data.
    - delta: int
        Number of elements to set to False at the start and end of each cluster.

    Returns:
    - np.ndarray
        Modified boolean array with unstable periods removed.
    """
    labels, num_c = label(arr)
    for cluster_id in range(1, num_c + 1):
        # Find indices belonging to the current cluster
        c_ix = np.where(labels == cluster_id)[0]
        # Set first two and last True values to False
        arr[c_ix[:delta]] = False  # First two
        arr[c_ix[-delta:]] = False  # Last
    return arr

# Extract indices of values within the desired range
def outlier_filter(p_batch, p_center, percentile=75):
    norm_array = np.linalg.norm((p_batch - p_center), axis=1)
    percentile_max = np.percentile(norm_array, percentile)
    indices = np.where(norm_array <= percentile_max)[0]
    p_batch_new = p_batch[indices]
    p_center_new = np.nanmedian(p_batch_new, axis=0)
    return p_batch_new, p_center_new


def interest_error(abs_err_x, abs_err_y, mov_ranges):
    # # Compute absolute errors for each movement type
    saccades_err_x = abs_err_x[mov_ranges['saccades']['H'][0]:mov_ranges['saccades']['H'][1]]
    saccades_err_y = abs_err_y[mov_ranges['saccades']['V'][0]:mov_ranges['saccades']['V'][1]]
    fixation_nystagmus_err_x = abs_err_x[mov_ranges['fixation_nystagmus']['H'][0]:mov_ranges['fixation_nystagmus']['H'][1]]
    fixation_nystagmus_err_y = abs_err_y[mov_ranges['fixation_nystagmus']['V'][0]:mov_ranges['fixation_nystagmus']['V'][1]]
    smooth_pursuit_err_x = abs_err_x[mov_ranges['smooth_pursuit']['H'][0]:mov_ranges['smooth_pursuit']['H'][1]]
    smooth_pursuit_err_y = abs_err_y[mov_ranges['smooth_pursuit']['V'][0]:mov_ranges['smooth_pursuit']['V'][1]]
    optokinetic_nystagmus_err_x = abs_err_x[mov_ranges['optokinetic_nystagmus']['H'][0]:mov_ranges['optokinetic_nystagmus']['H'][1]]
    optokinetic_nystagmus_err_y = abs_err_y[mov_ranges['optokinetic_nystagmus']['V'][0]:mov_ranges['optokinetic_nystagmus']['V'][1]]
    # Concatenate all 'x' errors and 'y' errors
    err_x_interest = np.concatenate([saccades_err_x, fixation_nystagmus_err_x,smooth_pursuit_err_x, optokinetic_nystagmus_err_x])
    err_y_interest = np.concatenate([saccades_err_y, fixation_nystagmus_err_y,smooth_pursuit_err_y, optokinetic_nystagmus_err_y])
    return err_x_interest, err_y_interest   



def gaze_comp_vis(subject_name, gaze_x_GT, gaze_y_GT, gaze_x_LG, gaze_y_LG, gaze_x_PL_angle, gaze_y_PL_angle, gaze_x_deepPL_angle, gaze_y_deepPL_angle, fps):
    t_max_x = min(gaze_x_GT.shape[0], gaze_x_LG.shape[0], gaze_x_PL_angle.shape[0], gaze_x_deepPL_angle.shape[0])
    t_max_y = min(gaze_y_GT.shape[0], gaze_y_LG.shape[0], gaze_y_PL_angle.shape[0], gaze_y_deepPL_angle.shape[0])
    # t = np.arange(0, t_max/fps, 1/fps)
    t_x = np.arange(0, t_max_x)
    t_x, tf_x = t_x[:-1], len(t_x) - 1
    t_y = np.arange(0, t_max_y)
    t_y, tf_y = t_y[:-1], len(t_y) - 1
    
    print(f"gaze_x_GT, survived tp: { np.sum(~np.isnan(gaze_x_GT))}")
    print(f"gaze_x_LG, survived tp: { np.sum(~np.isnan(gaze_x_LG))}")
    print(f"gaze_x_PL_angle, survived tp: { np.sum(~np.isnan(gaze_x_PL_angle))}")
    print(f"gaze_x_deepPL_angle, survived tp: { np.sum(~np.isnan(gaze_x_deepPL_angle))}")
    print(f"gaze_y_GT, survived tp: { np.sum(~np.isnan(gaze_y_GT))}")
    print(f"gaze_y_LG, survived tp: { np.sum(~np.isnan(gaze_y_LG))}")
    print(f"gaze_y_PL_angle, survived tp: { np.sum(~np.isnan(gaze_y_PL_angle))}")
    print(f"gaze_y_deepPL_angle, survived tp: { np.sum(~np.isnan(gaze_y_deepPL_angle))}")
   
    gaze_x_diff_LG = np.abs(gaze_x_GT[:tf_x] - gaze_x_LG[:tf_x])
    gaze_x_diff_PL_angle = np.abs(gaze_x_GT[:tf_x] - gaze_x_PL_angle[:tf_x])
    gaze_x_diff_deepPL_angle = np.abs(gaze_x_GT[:tf_x] - gaze_x_deepPL_angle[:tf_x])
    gaze_y_diff_LG = np.abs(gaze_y_GT[:tf_y] - gaze_y_LG[:tf_y])
    gaze_y_diff_PL_angle = np.abs(gaze_y_GT[:tf_y] - gaze_y_PL_angle[:tf_y])
    gaze_y_diff_deepPL_angle = np.abs(gaze_y_GT[:tf_y] - gaze_y_deepPL_angle[:tf_y])
    
    print(f"gaze_x_diff_LG, Median (min-max): {np.nanmedian(gaze_x_diff_LG)} ({np.nanmin(gaze_x_diff_LG)}-{np.nanmax(gaze_x_diff_LG)})")
    print(f"gaze_x_diff_PL, Median (min-max): {np.nanmedian(gaze_x_diff_PL_angle)} ({np.nanmin(gaze_x_diff_PL_angle)}-{np.nanmax(gaze_x_diff_PL_angle)})")
    print(f"gaze_x_diff_deepPL, Median (min-max): {np.nanmedian(gaze_x_diff_deepPL_angle)} ({np.nanmin(gaze_x_diff_deepPL_angle)}-{np.nanmax(gaze_x_diff_deepPL_angle)})")
    print(f"gaze_y_diff_LG, Median (min-max): {np.nanmedian(gaze_y_diff_LG)} ({np.nanmin(gaze_y_diff_LG)}-{np.nanmax(gaze_y_diff_LG)})")
    print(f"gaze_y_diff_PL, Median (min-max): {np.nanmedian(gaze_y_diff_PL_angle)} ({np.nanmin(gaze_y_diff_PL_angle)}-{np.nanmax(gaze_y_diff_PL_angle)})")
    print(f"gaze_y_diff_deepPL, Median (min-max): {np.nanmedian(gaze_y_diff_deepPL_angle)} ({np.nanmin(gaze_y_diff_deepPL_angle)}-{np.nanmax(gaze_y_diff_deepPL_angle)})")
    
    # Calculate Spearman correlation coefficient
    mask_LG_x = ~np.isnan(gaze_x_GT[:tf_x]) & ~np.isnan(gaze_x_LG[:tf_x])
    mask_PL_x = ~np.isnan(gaze_x_GT[:tf_x]) & ~np.isnan(gaze_x_PL_angle[:tf_x])
    mask_deepPL_x = ~np.isnan(gaze_x_GT[:tf_x]) & ~np.isnan(gaze_x_deepPL_angle[:tf_x])
    mask_LG_y = ~np.isnan(gaze_y_GT[:tf_y]) & ~np.isnan(gaze_y_LG[:tf_y])
    mask_PL_y = ~np.isnan(gaze_y_GT[:tf_y]) & ~np.isnan(gaze_y_PL_angle[:tf_y])
    mask_deepPL_y = ~np.isnan(gaze_y_GT[:tf_y]) & ~np.isnan(gaze_y_deepPL_angle[:tf_y])
    
    def correlation_calculation(mask, GT, comp_array, title):
        if np.sum(mask) > 0:
            correlation, p_value = spearmanr(GT[mask], comp_array[mask])
            print(f"Spearman correlation: {title}: {correlation:.3f}, p-value: {p_value:.3e}")
        else:
            print("Insufficient data to calculate correlation")
            
    correlation_calculation(mask_LG_x, gaze_x_GT[:tf_x], gaze_x_LG[:tf_x], '(gaze_x_GT vs gaze_x_LG)')
    correlation_calculation(mask_PL_x, gaze_x_GT[:tf_x], gaze_x_PL_angle[:tf_x], '(gaze_x_GT vs gaze_x_PL)')
    correlation_calculation(mask_deepPL_x, gaze_x_GT[:tf_x], gaze_x_deepPL_angle[:tf_x], '(gaze_x_GT vs gaze_x_deepPL)')
    correlation_calculation(mask_LG_y, gaze_y_GT[:tf_y], gaze_y_LG[:tf_y], '(gaze_x_GT vs gaze_y_LG)')
    correlation_calculation(mask_PL_y, gaze_y_GT[:tf_y], gaze_y_PL_angle[:tf_y], '(gaze_x_GT vs gaze_y_PL)')
    correlation_calculation(mask_deepPL_y, gaze_y_GT[:tf_y], gaze_y_deepPL_angle[:tf_y], '(gaze_x_GT vs gaze_y_deepPL)')
             
    # Create the figure
    fig = make_subplots(
        rows=2, cols=4, 
        column_widths=[0.7, 0.05, 0.2, 0.05],
        specs=[[{"colspan": 1}, None, {"colspan": 1, "type": "xy"}, None], 
               [{"colspan": 1}, None, {"colspan": 1, "type": "xy"}, None]],
        subplot_titles=("Horizontal Angular Movement", "Horizontal Error Distribution", "Vertical Angular Movement", "Vertical Error Distribution", "", ""),
        horizontal_spacing=0.02, 
        vertical_spacing=0.15
    )
        
    # Horizontal angular movement plot
    fig.add_trace(go.Scatter(x=t_x, y=gaze_x_LG[:tf_x], mode='lines', name='LG', line=dict(color='blue', width=0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_x, y=gaze_x_GT[:tf_x], mode='lines', name='GT', line=dict(color='red', width=0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_x, y=gaze_x_PL_angle[:tf_x], mode='lines', name='PL', line=dict(color='green', width=0.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_x, y=gaze_x_deepPL_angle[:tf_x], mode='lines', name='deepPL', line=dict(color='purple', width=0.5)), row=1, col=1)

    # Vertical angular movement plot
    fig.add_trace(go.Scatter(x=t_y, y=gaze_y_LG[:tf_y], mode='lines', name='LG', line=dict(color='blue', width=0.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_y, y=gaze_y_GT[:tf_y], mode='lines', name='GT', line=dict(color='red', width=0.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_y, y=gaze_y_PL_angle[:tf_y], mode='lines', name='PL', line=dict(color='green', width=0.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=t_y, y=gaze_y_deepPL_angle[:tf_y], mode='lines', name='deepPL', line=dict(color='purple', width=0.5)), row=2, col=1)

    # Function to add CDF plots
    def add_cdf_plot(fig, data, row, col, name, color):
        data_with_nan = data.copy()
        data_with_nan[np.isnan(data_with_nan)] = 1000  # Set NaN values to 1000

        # Clip data to range 0-10 degrees
        clipped_data = np.clip(data_with_nan, 0, 10)

        # Calculate CDF
        sorted_data = np.sort(clipped_data)
        cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        fig.add_trace(go.Scatter(x=sorted_data, y=cdf, mode='lines', name=f'{name} CDF', line=dict(color=color, width=1)), row=row, col=col)

    add_cdf_plot(fig, gaze_x_diff_LG, 1, 3, 'LG Error', 'blue')
    add_cdf_plot(fig, gaze_x_diff_PL_angle, 1, 3, 'PL Error', 'green')
    add_cdf_plot(fig, gaze_x_diff_deepPL_angle, 1, 3, 'deepPL Error', 'purple')

    add_cdf_plot(fig, gaze_y_diff_LG, 2, 3, 'LG Error', 'blue')
    add_cdf_plot(fig, gaze_y_diff_PL_angle, 2, 3, 'PL Error', 'green')
    add_cdf_plot(fig, gaze_y_diff_deepPL_angle, 2, 3, 'deepPL Error', 'purple')

    # Update layout
    fig.update_layout(
        height=800,
        width=1500,
        title_text= f"Gaze Angular Movements - {subject_name}",
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=True
    )

    # Update x and y axes for black ticks and box, without grid lines
    axis_common_settings = dict(
        showgrid=False,
        zeroline=False,
        linecolor='black',
        showline=True,
        ticks="inside",
        ticklen=5,
        tickwidth=2,
        mirror=True,
    )

    fig.update_xaxes(title_text="Time [s]", **axis_common_settings, row=1, col=1)
    fig.update_yaxes(title_text="Horizontal Angular Movement (degree)", range=[-30, 30], row=1, col=1, **axis_common_settings)
    fig.update_xaxes(title_text="Time [s]", **axis_common_settings, row=2, col=1)
    fig.update_yaxes(title_text="Vertical Angular Movement (degree)", range=[-30, 30], row=2, col=1, **axis_common_settings)
    fig.update_xaxes(title_text="Error [°]", range=[0, 5], **axis_common_settings, row=1, col=3)
    fig.update_yaxes(title_text="Cumulative Probability", range=[0, 1], **axis_common_settings, row=1, col=3)
    fig.update_xaxes(title_text="Error [°]", range=[0, 5], **axis_common_settings, row=2, col=3)
    fig.update_yaxes(title_text="Cumulative Probability", range=[0, 1], **axis_common_settings, row=2, col=3)

    fig.show()
