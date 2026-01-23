import numpy as np
import torch
'''
PL Coordinate System:    
          ^ x                 
          |                  
     z<---⨁ y                      
   
LG Coordinate System:
     z<---⊙ y
          |
          v x
'''
def _normalize_vectors(N: np.ndarray | torch.Tensor, eps: float = 1e-8)-> np.ndarray | torch.Tensor:
    """
    Normalize 3D vectors row-wise.

    Supports both numpy arrays and torch tensors.

    - Input shape: (B, 3)
    - Each vector is normalized to unit length
    - Zero or near-zero vectors remain zero (prevents NaNs)

    Parameters
    ----------
    N : np.ndarray or torch.Tensor
        Batch of 3D vectors.
    eps : float
        Small threshold to detect near-zero vectors.

    Returns
    -------
    Nn : same type as input
        Normalized vectors with the same shape as N.
    """

    # -------- NumPy implementation --------
    if isinstance(N, np.ndarray):
        norms = np.linalg.norm(N, axis=1, keepdims=True)  # (B,1)
        mask = norms > eps                               # valid vectors
        Nn = np.zeros_like(N)
        Nn[mask[:, 0]] = N[mask[:, 0]] / norms[mask, None]
        return Nn

    # -------- PyTorch implementation --------
    elif torch.is_tensor(N):
        norms = torch.linalg.norm(N, dim=1, keepdim=True)  # (B,1)
        mask = norms > eps
        Nn = torch.zeros_like(N)
        Nn[mask.squeeze(1)] = N[mask.squeeze(1)] / norms[mask, None]
        return Nn

    else:
        raise TypeError("N must be a numpy array or torch tensor")


def cart2sph_batch_PL(N: np.ndarray | torch.Tensor)-> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Convert Cartesian vectors to spherical coordinates
    using the Pupil Labs (PL) convention.

    Input:
    ------
    N : (B, 3) numpy array or torch tensor
        Cartesian direction vectors.

    Output:
    -------
    phi : azimuth angle in degrees
    theta : elevation angle in degrees

    Notes:
    ------
    - Vectors are normalized internally.
    - Zero vectors return (0, 0).
    - Coordinate convention follows Pupil Labs gaze definition.
    """

    N = _normalize_vectors(N)

    # -------- NumPy implementation --------
    if isinstance(N, np.ndarray):
        phi = np.arctan2(N[:, 2], N[:, 0]) + np.pi / 2
        theta = np.arccos(np.clip(N[:, 1], -1.0, 1.0)) - np.pi / 2
        return np.rad2deg(phi), np.rad2deg(theta)

    # -------- PyTorch implementation --------
    else:
        phi = torch.atan2(N[:, 2], N[:, 0]) + torch.pi / 2
        theta = torch.acos(torch.clamp(N[:, 1], -1.0, 1.0)) - torch.pi / 2
        return torch.rad2deg(phi), torch.rad2deg(theta)


def cart2sph_batch(N: np.ndarray | torch.Tensor)-> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Convert Cartesian vectors to spherical coordinates
    using the legacy 3DeepVOG convention.

    Input:
    ------
    N : (B, 3) numpy array or torch tensor
        Cartesian direction vectors.

    Output:
    -------
    phi : azimuth angle in degrees
    theta : elevation angle in degrees

    Notes:
    ------
    - Uses arctan2-based elevation definition.
    - Sign conventions differ from PL version.
    - Vectors are normalized internally.
    """

    N = _normalize_vectors(N)

    # -------- NumPy implementation --------
    if isinstance(N, np.ndarray):
        phi = -(np.arctan2(N[:, 2], N[:, 0]) + np.pi / 2)
        theta = -(np.arctan2(N[:, 2], N[:, 1]) - np.pi / 2)
        return np.rad2deg(phi), np.rad2deg(theta)

    # -------- PyTorch implementation --------
    else:
        phi = -(torch.atan2(N[:, 2], N[:, 0]) + torch.pi / 2)
        theta = -(torch.atan2(N[:, 2], N[:, 1]) - torch.pi / 2)
        return torch.rad2deg(phi), torch.rad2deg(theta)