import math
import cv2
import numpy as np
import torch
from kornia.geometry.conversions import axis_angle_to_rotation_matrix


def to_torch(x, device=None, dtype=None):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device=device, dtype=dtype)
    elif isinstance(x, (float, int)):
        return torch.tensor(x, device=device, dtype=dtype)
    elif isinstance(x, (list, tuple)):
        # try numeric tensor, otherwise recurse
        try:
            return torch.tensor(x, device=device, dtype=dtype)
        except Exception:
            return [to_torch(v, device, dtype) for v in x]
    elif isinstance(x, dict):
        return {k: to_torch(v, device, dtype) for k, v in x.items()}
    else:
        return x  # leave strings, None, objects unchanged
    
def to_numpy(x):
    """
    Recursively convert torch tensors to numpy / python types.
    Mirrors `to_torch`.
    """
    if isinstance(x, torch.Tensor):
        # scalar tensor → python scalar
        if x.ndim == 0:
            return x.detach().cpu().item()
        # array tensor → numpy array
        return x.detach().cpu().numpy()

    elif isinstance(x, np.ndarray):
        return x

    elif isinstance(x, (float, int, str, bool)) or x is None:
        return x

    elif isinstance(x, (list, tuple)):
        return type(x)(to_numpy(v) for v in x)

    elif isinstance(x, dict):
        return {k: to_numpy(v) for k, v in x.items()}

    else:
        # leave objects unchanged (cv2 images, classes, etc.)
        return x

def projection(P, fcl, img_size = (240, 320), eps=1e-6):
    """
    Projects 3D points to 2D using focal lengths.
    Works for both torch.Tensor and np.ndarray.

    Args:
        P: torch.Tensor or np.ndarray [..., 3]
        fcl: scalar
        img_size: (H, W)
        eps: numerical stability

    Returns:
        Same type as input (torch.Tensor or np.ndarray) [..., 2]
    """
    # Determine how many extra dims to expand into (e.g., [B, N, 3, T, U])
    # extra_dims = P.dim() - 3  # because P is at least [B, N, 3]
    # expand_shape = [1] * extra_dims  # e.g., [1, 1] if extra dims are T and U
    cx, cy = (img_size[1]-1)/2, (img_size[0]-1)/2
    is_torch = torch.is_tensor(P)
    z = P[..., 2]
    z = z.clamp_min(eps) if is_torch else np.clip(z, eps, None)
    x = P[..., 0] / z * fcl + cx
    y = P[..., 1] / z * fcl + cy

    return (torch.stack((x, y), -1) if is_torch
            else np.stack((x, y), -1))



def PL2normDict_batch(results_3d: list[dict]) -> dict:
    B = len(results_3d)

    timestamp = np.zeros(B, dtype=np.float32)
    c_eye = np.zeros((B, 3), dtype=np.float32)
    r_eye = np.zeros(B, dtype=np.float32)
    c_eye2d = np.zeros((B, 2), dtype=np.float32)
    r_eye2d = np.zeros(B, dtype=np.float32)
    c_pupil = np.zeros((B, 3), dtype=np.float32)
    gaze = np.zeros((B, 3), dtype=np.float32)
    hor = np.zeros(B, dtype=np.float32)
    ver = np.zeros(B, dtype=np.float32)
    r_pupil = np.zeros(B, dtype=np.float32) 
    diameter_3d = np.zeros(B, dtype=np.float32)
    location = np.zeros((B, 2), dtype=np.float32)
    diameter = np.zeros(B, dtype=np.float32)
    confidence = np.zeros(B, dtype=np.float32)
    model_confidence = np.zeros(B, dtype=np.float32)
    norm_pos = np.zeros((B, 3), dtype=np.float32)
    entpup_el = np.zeros((B, 5), dtype=np.float32)

    for i, r in enumerate(results_3d):
        ps = r.get("projected_sphere", {})
        sp = r.get("sphere", {})
        c3 = r.get("circle_3d", {})
        el = r.get("ellipse", {})

        timestamp[i] = float(r.get("timestamp", 0.0))

        c_eye[i] = sp.get("center", (0.0, 0.0, 0.0))
        r_eye[i] = float(sp.get("radius", 0.0))

        c_eye2d[i] = ps.get("center", (0.0, 0.0))
        if "axes" in ps:
            r_eye2d[i] = np.mean(ps["axes"]) / 2.0

        c_pupil[i] = c3.get("center", (0.0, 0.0, 0.0))
        gaze[i] = c3.get("normal", (0.0, 0.0, 0.0))
        r_pupil[i] = float(c3.get("radius", 0.0))

        hor[i] = float(r.get("phi", 0.0))
        ver[i] = float(r.get("theta", 0.0))

        diameter_3d[i] = float(r.get("diameter_3d", 0.0))
        location[i] = r.get("location", (0.0, 0.0))
        diameter[i] = float(r.get("diameter", 0.0))
        confidence[i] = float(r.get("confidence", 0.0))
        model_confidence[i] = float(r.get("model_confidence", 0.0))
        norm_pos[i] = r.get("norm_pos", (0.0, 0.0, 0.0))

        entpup_el[i] = np.array([
            np.deg2rad(float(el.get("angle", 0.0))),
            *el.get("center", (0.0, 0.0)),
            *(np.asarray(el.get("axes", (0.0, 0.0)), dtype=np.float32) / 2.0),
        ], dtype=np.float32)

    return {
        "timestamp": timestamp,
        "c_eye": c_eye,
        "r_eye": r_eye,
        "c_eye2d": c_eye2d,
        "r_eye2d": r_eye2d,
        "c_pupil": c_pupil,
        "gaze": gaze,
        "hor": hor,
        "ver": ver,
        "r_pupil": r_pupil,
        "entpup_el": entpup_el,
        "diameter_3d": diameter_3d,
        "location": location,
        "diameter": diameter,
        "confidence": confidence,
        "model_confidence": model_confidence,
        "norm_pos": norm_pos,
    }


def calc_model_iris_mask(self, result_3d):
    theta = np.linspace(0, 2 * np.pi, 100)
    xc, yc, zc = result_3d['circle_3d']['center']
    n = np.array(result_3d['circle_3d']['normal'])
    n /= np.linalg.norm(n)
    ref = np.array([0, 0, 1]) if np.abs(n[2]) < 0.99 else np.array([1, 0, 0])
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    r = 6.0
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    X_3d = xc + r * (cos_t * u[0] + sin_t * v[0])
    Y_3d = yc + r * (cos_t * u[1] + sin_t * v[1])
    Z_3d = zc + r * (cos_t * u[2] + sin_t * v[2])
    vid_w, vid_h = self.args['resolution']
    x_2d = self.fcl_px * (X_3d / Z_3d) + vid_w * 0.5
    y_2d = self.fcl_px * (Y_3d / Z_3d) + vid_h * 0.5
    # STEP 3: Rasterize to mask
    H, W = self.args['resolution'][1], self.args['resolution'][0]
    iris_proj_mask = np.zeros((H, W), dtype=np.uint8)
    pts = np.stack([x_2d, y_2d], axis=1).astype(np.int32).reshape((-1, 1, 2))
    cv2.fillPoly(iris_proj_mask, [pts], color=1)
    return iris_proj_mask.astype(bool)



def rend_params(out_dict: dict) -> dict:
    """
    :param out_dict: 
    should contain at least:
        'r_pupil': torch.Size([B, 1])
        'r_iris':  torch.Size([B, 1])   
        'r_cornea': torch.Size([B, 1])
        'c_cornea' or 'c_eye' or 'c_pupil': torch.Size([B, 3])
        'gaze' or 'R': torch.Size([B, 3])
        'img_size': (W, H)
        'fcl_mm':  scaler

    returns: updated out_dict with computed parameters
    """
    B, _   = out_dict['r_iris'].shape
    device = out_dict['r_iris'].device
    r_iris   = out_dict['r_iris']
    r_cornea = out_dict['r_cornea']
    img_size = out_dict['img_size']
    r_eye = out_dict['r_eye']
    r_pupil  = out_dict['r_pupil']
    gaze = out_dict["gaze"]
    torsion = out_dict["torsion"]
    fpx = out_dict['fcl_px']
    out_dict["L_p"]  = (L_p  := torch.sqrt((r_eye**2 - r_iris**2).clamp_min_(1e-6)))
    out_dict["L_ec"] = (L_ec := L_p - torch.sqrt((r_cornea**2 - r_iris**2).clamp_min_(1e-6)))
    c_eye    = out_dict["c_eye"]
    c_pupil  = out_dict["c_pupil"]
    out_dict["c_cornea"] = (c_cornea := c_eye + L_ec * gaze)

    # if np.isnan(c_eye).any() or np.isnan(c_pupil).any(): return None
    # gaze_vec_rev = torch.clone(gaze_vec)*torch.tensor([1.0, 1.0, -1.0], device=gaze_vec.device)  # flip z to match camera coordinates
    c_eye2d = out_dict.get("c_eye2d", projection(c_eye, fpx, img_size))
    c_pupil2d = out_dict.get("c_pupil2d", projection(c_pupil, fpx, img_size))
    gaze2d = out_dict.get("gaze2d", projection(gaze, fpx, img_size))
    refpup_el, ok = circle2ellipse(c_pupil, gaze, r_pupil, fpx, img_size)
    iris_el, ok = circle2ellipse(c_pupil, gaze, r_iris, fpx, img_size)

    out_dict.update({
        'c_eye2d': c_eye2d,
        'c_pupil2d': c_pupil2d,
        'gaze2d': gaze2d,
        'refpup_el': refpup_el,   
        'iris_el': iris_el,
    })

    num_grids = 25
    d_lim_cornea = L_p - L_ec
    limbus_theta_eye, limbus_theta_cornea = torch.arccos(L_p / r_eye), torch.arccos(d_lim_cornea / r_cornea)
    large_meshes = gen_sphere_mesh(r_eye, num_grids, insert_theta=limbus_theta_eye)
    small_meshes = gen_sphere_mesh(r_cornea, num_grids, insert_theta=limbus_theta_cornea)

    # build R aligned+torsion
    # ref_axis = torch.tensor([0,0,-1], dtype=torch.float32, device=device)
    R_comb  = build_rotation_matrices(gaze, torsion).to(device=device, dtype=torch.float32)
    # R_comb = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).repeat(B, 1, 1)

    # rotate & shift
    large_meshes = rotate_and_shift(large_meshes, c_eye,   R_comb)
    small_meshes = rotate_and_shift(small_meshes, c_cornea,R_comb)

    # Expand to match temporal dimension T
    eps = 1e-6  # or 1e-6, tune
    r_cornea2 = (r_cornea.view(-1)**2)[:, None, None]
    r_eye2    = (r_eye.view(-1)**2)[:, None, None]

    d_large = ((large_meshes - c_cornea[:,None,None,:])**2).sum(-1)
    d_small = ((small_meshes - c_eye[:,None,None,:])**2).sum(-1)
    large_mask = d_large < (r_cornea2 - eps)          # inside cornea => nan
    small_mask = d_small > (r_eye2 - eps)             # outside eyeball => keep (like your current logic)
    large_meshes = torch.where(large_mask[...,None], torch.nan, large_meshes)
    small_meshes = torch.where(small_mask[...,None], small_meshes, torch.nan)

    large_coords_2d = projection(large_meshes, fpx, img_size)
    small_coords_2d = projection(small_meshes, fpx, img_size)
    vis_dicts = {
        "eyeball_mesh": large_meshes, 
        "corneaball_mesh": small_meshes,
        "eyeball_mesh_2d": large_coords_2d, 
        "corneaball_mesh_2d": small_coords_2d,
    }
    out_dict = {**out_dict, **vis_dicts}
    return dict(out_dict)   # shallow copy (keys point to same tensors, but dict not shared)



def gen_sphere_mesh(r, n_grids=25, insert_theta=None, dedup_eps=0.0):
    """
    Generate a batched 3D spherical mesh grid with optional per-sample theta insertions.

    Args:
        r:              [B, 1] or [B] radius (device/dtype drive the outputs)
        n_grids:        number of uniform theta/phi divisions
        insert_theta:   optional angles to insert (radians). Accepted shapes:
                         - [B, 1, 1]
                         - [B, N, 1]
                         - [B, N]
                         - [B]      (N = #inserts per sample)
        dedup_eps:      if >0, masks near-duplicate thetas within eps (per sample) as NaN

    Returns:
        mesh: [B, Θ, Φ, 3], Θ = n_grids (+ K if inserts provided), Φ = n_grids
    """
    # ---- radius shape handling ----
    if r.ndim == 2:
        B, C = r.shape
        assert C == 1, f"r must be [B,1] or [B], got {r.shape}"
        r = r.squeeze(-1)          # [B]
    elif r.ndim == 1:
        B = r.shape[0]
    else:
        raise ValueError(f"Unsupported r shape {tuple(r.shape)}, expected [B,1] or [B]")

    device = r.device
    G = int(n_grids)

    # Base 1D grids
    theta_base = torch.linspace(0.0, math.pi, G, device=device)       # [G]
    phi_base   = torch.linspace(0.0, 2.0 * math.pi, G, device=device) # [G]

    # ---- normalize insert_theta to [B, K] or None ----
    if insert_theta is not None:
        ins = insert_theta.to(device)

        if ins.ndim == 3:
            # [B, N, 1] or [B, 1, 1]
            b, n, c = ins.shape
            assert b == B, f"insert_theta B mismatch: {b} vs {B}"
            assert c == 1, f"Last dim of insert_theta must be 1, got {c}"
            ins = ins.squeeze(-1)          # [B, N]

        elif ins.ndim == 2:
            # [B, N]
            b, n = ins.shape
            assert b == B, f"insert_theta B mismatch: {b} vs {B}"
            # already [B, N]

        elif ins.ndim == 1:
            # [B] -> one insert per sample
            assert ins.shape[0] == B, "insert_theta B mismatch"
            ins = ins.unsqueeze(1)         # [B, 1]

        else:
            raise ValueError(f"Unsupported insert_theta ndim {ins.ndim}")

        K_ins = ins.shape[1]

        # Build per-sample theta by concatenation: [B, G+K]
        theta_vals = torch.cat([
            theta_base.view(1, G).expand(B, G),   # [B, G]
            ins                                   # [B, K_ins]
        ], dim=-1)                                # [B, G+K_ins]

        # Sort per-sample
        theta_vals, _ = theta_vals.sort(dim=-1)

        # Optional near-duplicate masking (per-sample, within eps)
        if dedup_eps > 0.0:
            diffs = theta_vals.diff(dim=-1, prepend=theta_vals[:, :1])
            keep  = diffs > dedup_eps
            theta_vals = torch.where(keep, theta_vals, torch.nan)  # masked duplicates
    else:
        theta_vals = theta_base.view(1, G).expand(B, G)            # [B, G]

    # ---- make 2D grids via broadcasting ----
    # θ: [B, Θ, 1], φ: [1, 1, Φ] -> broadcast to [B, Θ, Φ]
    theta = theta_vals.unsqueeze(-1)                               # [B, Θ, 1]
    phi   = phi_base.view(1, 1, G)                                 # [1, 1, Φ]

    # radius -> [B, 1, 1]
    r_b11 = r.view(B, 1, 1)

    # sphere (camera looks along -Z)
    sinθ = torch.sin(theta)
    cosθ = torch.cos(theta)
    cosφ = torch.cos(phi)
    sinφ = torch.sin(phi)

    X = r_b11 * sinθ * cosφ        # [B, Θ, Φ]
    Y = r_b11 * sinθ * sinφ        # [B, Θ, Φ]
    Z = (r_b11 * cosθ).expand_as(X)

    mesh = torch.stack([X, Y, Z], dim=-1)  # [B, Θ, Φ, 3]
    return mesh



import torch

def rot_to_align_with_z_torch(v: torch.Tensor, eps=1e-8) -> torch.Tensor:
    """
    v: (B,3) vectors
    returns: (B,3,3) rotation matrices that rotate v -> +z
    Matches your NumPy rot_to_align_with_z behavior including special cases.
    """
    B = v.shape[0]
    device, dtype = v.device, v.dtype

    v = v / (v.norm(dim=-1, keepdim=True).clamp_min(eps))
    z = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype).view(1, 3).expand(B, 3)

    cos_angle = (v * z).sum(dim=-1).clamp(-1.0, 1.0)  # (B,)
    angle = torch.acos(cos_angle)                      # (B,)

    # axis = cross(v, z)
    axis = torch.cross(v, z, dim=-1)                   # (B,3)
    axis_norm = axis.norm(dim=-1, keepdim=True)

    # Identity where v ~ +z
    I = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, 3, 3)
    is_plus = (angle < 1e-7)

    # 180 deg around x-axis where v ~ -z (same as your numpy special case)
    is_minus = (torch.abs(angle - torch.pi) < 1e-7)
    Rx_pi = I.clone()
    Rx_pi[:, 1, 1] = -1
    Rx_pi[:, 2, 2] = -1

    # General case: Rodrigues from axis-angle (axis normalized)
    axis_unit = axis / axis_norm.clamp_min(eps)
    aa = axis_unit * angle.unsqueeze(-1)  # (B,3)

    R = axis_angle_to_rotation_matrix(aa)  # you already have this

    # Stitch cases
    R = torch.where(is_plus.view(B, 1, 1), I, R)
    R = torch.where(is_minus.view(B, 1, 1), Rx_pi, R)
    return R


def build_rotation_matrices(gaze_vec: torch.Tensor, torsion_angle: torch.Tensor) -> torch.Tensor:
    """
    gaze_vec: (B,3) unit or non-unit
    torsion_angle: (B,1) radians
    returns: (B,3,3) R_total = R_torsion @ R_gaze (NumPy equivalent)
    """
    eps = 1e-8
    g = gaze_vec / (gaze_vec.norm(dim=-1, keepdim=True).clamp_min(eps))

    # rot_to_align_with_z(g) returns g -> +z, so transpose is +z -> g
    gaze_align_rot = rot_to_align_with_z_torch(g)      # (B,3,3) : g -> z
    R_gaze = gaze_align_rot.transpose(-1, -2)          # (B,3,3) : z -> g

    # torsion around gaze axis (same as scipy Rotation.from_rotvec(torsion*g))
    aa2 = g * torsion_angle                            # (B,3)
    R_torsion = axis_angle_to_rotation_matrix(aa2)     # (B,3,3)

    return R_torsion @ R_gaze


import torch

def rotate_and_shift(meshes: torch.Tensor, centers: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    meshes:  (B,Θ,Φ,3) OR (B,3,Θ,Φ)
    centers: (B,3) or (B,1,3)
    R:       (B,3,3)
    returns: (B,Θ,Φ,3)
    """
    B = meshes.shape[0]

    centers = centers.view(B, -1)
    if centers.shape[1] != 3:
        raise ValueError(f"centers must be (B,3) or (B,1,3), got {centers.shape}")

    if meshes.shape[-1] == 3:
        # (B,Θ,Φ,3) -> (B,Θ*Φ,3)
        T, U = meshes.shape[1], meshes.shape[2]
        pts = meshes.reshape(B, T * U, 3)
        # row-vectors: p' = p @ R^T
        pts_rot = torch.bmm(pts, R.transpose(1, 2))   # (B,Θ*Φ,3)
        rotated = pts_rot.reshape(B, T, U, 3)

    elif meshes.shape[1] == 3:
        # (B,3,Θ,Φ) -> rotate as column-vectors: p' = R @ p
        rotated = torch.einsum("bij,bjtu->bitu", R, meshes).permute(0, 2, 3, 1)  # (B,Θ,Φ,3)

    else:
        raise ValueError(f"meshes must be (B,Θ,Φ,3) or (B,3,Θ,Φ), got {meshes.shape}")

    return rotated + centers[:, None, None, :]



from torch.nn import functional as F
# --- cheap prior (no refraction) ---
def circle2ellipse(
    c_pupil,        # (B,3) camera coords
    gaze,           # (B,3) plane normal
    r_pupil,        # (B,) or (B,1)
    fcl,            # scalar or 0-D tensor (pixels)
    img_size,       # (H, W)
    eps=1e-12,
    internal_dtype=torch.float64,
):
    """
    Exact pinhole projection of a 3D circle to an image ellipse.

    Returns:
      E, ok
      (B,5) [cx,cy,a,b,θ] otherwise
      ok: (B,) boolean validity mask
    """
    B = c_pupil.shape[0]
    dev = c_pupil.device
    out_dtype = c_pupil.dtype
    Himg, Wimg = img_size

    to_id = lambda t: t.to(device=dev, dtype=internal_dtype)

    # Promote to fp64 internally
    c_p  = to_id(c_pupil)                          # (B,3)
    gz   = F.normalize(to_id(gaze), dim=-1)        # (B,3)

    # focal length as scalar tensor (pixels)
    if isinstance(fcl, (int, float)):
        fcl_id = torch.tensor(float(fcl), device=dev, dtype=internal_dtype)
    else:
        fcl_id = to_id(fcl).reshape(())

    # radius -> (B,)
    if r_pupil.ndim == 2 and r_pupil.shape[-1] == 1:
        rp = to_id(r_pupil.squeeze(-1))
    else:
        rp = to_id(r_pupil)

    eps_id = torch.tensor(float(eps), dtype=internal_dtype, device=dev)

    # --- intrinsics (consistent with your projection: (W-1)/2, (H-1)/2) ---
    cx0 = (Wimg - 1) * 0.5
    cy0 = (Himg - 1) * 0.5
    K = torch.tensor([[fcl_id, 0.0,   cx0],
                      [0.0,    fcl_id, cy0],
                      [0.0,    0.0,    1.0]],
                     device=dev, dtype=internal_dtype).expand(B, -1, -1)

    # --- plane ONB (u,v,n) ---
    ax = torch.tensor([1.,0.,0.], device=dev, dtype=internal_dtype).expand_as(gz)
    ay = torch.tensor([0.,1.,0.], device=dev, dtype=internal_dtype).expand_as(gz)
    aux = torch.where((gz[...,0].abs() >= 0.9)[...,None], ay, ax)  # choose aux ≠ n
    u = F.normalize(torch.cross(gz, aux, dim=-1), dim=-1)         # (B,3)
    v = F.normalize(torch.cross(gz, u,   dim=-1), dim=-1)         # (B,3)

    # --- homography H: [s,t,1]^T (plane) -> image ---
    # X = c_p + u*s + v*t; x ~ K X  =>  H = K [u v c_p]
    H_3d = torch.stack([
        torch.stack([u[:,0], v[:,0], c_p[:,0]], dim=-1),
        torch.stack([u[:,1], v[:,1], c_p[:,1]], dim=-1),
        torch.stack([u[:,2], v[:,2], c_p[:,2]], dim=-1),
    ], dim=1)                                                   # (B,3,3)
    H = torch.einsum('bij,bjk->bik', K, H_3d)                   # (B,3,3)

    # Guard: det(H) shouldn't be tiny
    detH = torch.linalg.det(H)
    goodH = detH.abs() > eps_id

    # --- primal circle conic in plane coords: s^2 + t^2 = r^2
    #     C_p = diag(1,1,-r^2) up to scale
    C_p = torch.zeros(B, 3, 3, device=dev, dtype=internal_dtype)
    C_p[:,0,0] = 1.0
    C_p[:,1,1] = 1.0
    C_p[:,2,2] = -(rp * rp).clamp_min(eps_id)

    # --- image primal conic: C_img ∝ H^{-T} C_p H^{-1} ---
    H_inv = torch.empty_like(H)
    ok = torch.isfinite(H).all(dim=(-1, -2)) & (torch.linalg.det(H).abs() > 1e-12)

    if ok.any():
        H_inv[ok] = torch.linalg.inv(H[ok])

    if (~ok).any():
        H_inv[~ok] = torch.linalg.pinv(H[~ok])

    H_inv_T = H_inv.transpose(1,2)
    Ci = torch.einsum('bij,bjk,bkl->bil', H_inv_T, C_p, H_inv)
    Ci = 0.5 * (Ci + Ci.transpose(1,2))          # enforce symmetry

    # --- scale-normalize to keep numbers conditioned ---
    A = Ci[:,0,0]; Bc = Ci[:,0,1]; Cq = Ci[:,1,1]
    s_norm = torch.stack([A, 2*Bc, Cq], dim=-1).norm(dim=-1).clamp_min(eps_id)
    Ci = Ci / s_norm[:,None,None]

    # --- decode center/axes/angle (same convention as your other code) ---
    # [x y 1] Ci [x y 1]^T = 0  ->  A x^2 + 2B xy + C y^2 + 2D x + 2E y + F = 0
    A = Ci[:,0,0]; B = Ci[:,0,1]; Cq = Ci[:,1,1]
    D = Ci[:,0,2]; E = Ci[:,1,2]; Fv = Ci[:,2,2]

    # Center: solve [2A 2B; 2B 2C] [cx; cy] = [-2D; -2E]
    Q2  = torch.stack([torch.stack([2*A, 2*B], dim=-1),
                       torch.stack([2*B, 2*Cq], dim=-1)], dim=1)      # (B,2,2)
    rhs = torch.stack([-2*D, -2*E], dim=-1)[...,None]                  # (B,2,1)
    cxcy = torch.linalg.lstsq(Q2, rhs, rcond=None).solution.squeeze(-1)
    cx, cy = cxcy[:,0], cxcy[:,1]

    # Evaluate F at center
    F0 = A*cx*cx + 2*B*cx*cy + Cq*cy*cy + 2*D*cx + 2*E*cy + Fv

    # Eigen-decomp of quadratic part (Q2/2 gives the actual A,C in rotated basis)
    evals, evecs = torch.linalg.eigh(Q2 / 2)

    lam1 = evals[:,0].abs().clamp_min(eps_id)
    lam2 = evals[:,1].abs().clamp_min(eps_id)
    rad2_major = (F0.abs() / lam1).clamp_min(eps_id)
    rad2_minor = (F0.abs() / lam2).clamp_min(eps_id)
    a = torch.sqrt(rad2_major)
    b = torch.sqrt(rad2_minor)

    # Ensure a >= b and choose matching eigenvector for θ
    swap = b > a
    a, b = torch.where(swap, b, a), torch.where(swap, a, b)
    maj_vec = torch.where(swap.unsqueeze(-1), evecs[:,:,1], evecs[:,:,0])  # (B,2)
    theta = torch.atan2(maj_vec[:,1], maj_vec[:,0])
    theta = ((theta + torch.pi/2) % torch.pi) - torch.pi/2                 # wrap to (-π/2, π/2]

    # Validity mask
    same_sign = torch.sign(evals[:,0]) == torch.sign(evals[:,1])
    finite_ok = torch.isfinite(a) & torch.isfinite(b) & torch.isfinite(cx) & torch.isfinite(cy)
    ok = finite_ok & same_sign & (F0.abs() > eps_id) & goodH

    E_out_id = torch.stack([theta, cx, cy, a, b], dim=-1)

    return E_out_id.to(out_dtype), ok.to(torch.bool)



