import numpy as np
import torch

# The whole unprojection algorithm is invented by Safaee-Rad et al. 1992, see https://ieeexplore.ieee.org/document/163786
# This python script is a re-implementation of Safaee-Rad et al.'s works


def gen_cone_co(alpha, beta, gamma, a_prime, h_prime, b_prime, g_prime, f_prime, d_prime):
    gamma_square = np.power(gamma,2)
    a = gamma_square * a_prime
    b = gamma_square * b_prime
    c = a_prime * np.power(alpha,2) + 2 * h_prime * alpha * beta + b_prime * np.power(beta,2) + 2 * g_prime * alpha + 2 * f_prime * beta + d_prime
    d = gamma_square * d_prime
    f = -gamma * (b_prime * beta + h_prime * alpha +f_prime)
    g = -gamma * (h_prime * beta + a_prime * alpha + g_prime)
    h = gamma_square * h_prime
    u = gamma_square * g_prime
    v = gamma_square * f_prime
    w = -gamma * (f_prime * beta + g_prime * alpha + d_prime)
    return a,b,c,d,f,g,h,u,v,w

def gen_cone_co_torch(alpha, beta, gamma, a_prime, h_prime, b_prime, g_prime, f_prime, d_prime):
    gamma_square = torch.pow(gamma,2)
    a = gamma_square * a_prime
    b = gamma_square * b_prime
    c = a_prime * torch.pow(alpha,2) + 2 * h_prime * alpha * beta + b_prime * torch.pow(beta,2) + 2 * g_prime * alpha + 2 * f_prime * beta + d_prime
    d = gamma_square * d_prime
    f = -gamma * (b_prime * beta + h_prime * alpha +f_prime)
    g = -gamma * (h_prime * beta + a_prime * alpha + g_prime)
    h = gamma_square * h_prime
    u = gamma_square * g_prime
    v = gamma_square * f_prime
    w = -gamma * (f_prime * beta + g_prime * alpha + d_prime)
    return a,b,c,d,f,g,h,u,v,w

'''
Safaee-Rad, 1992 (8)

'''
def gen_rotmat_co(lamb, a,b,g,f,h):
    t1 = (b-lamb)*g - f*h
    t2 = (a - lamb)*f - g*h
    t3 = -(a-lamb)*(t1/t2)/g - (h/g)
    m = 1/(np.sqrt(1+np.power((t1/t2),2)+np.power(t3,2)))
    l = (t1/t2)*m
    n = t3*m
    return l, m, n

import torch

def gen_rotmat_co_batch(lamb_batch, a, b, g, f, h):
    """
    Vectorized version of gen_rotmat_co for batches.
    Computes the components of the rotation matrix for a batch of eigenvalues.

    Args:
        lamb_batch: (batch_size,) tensor of eigenvalues.
        a, b, g, f, h: Scalar parameters.

    Returns:
        l_batch: (batch_size,) tensor of l components.
        m_batch: (batch_size,) tensor of m components.
        n_batch: (batch_size,) tensor of n components.
    """
    # Compute intermediate terms
    t1 = (b - lamb_batch) * g - f * h
    t2 = (a - lamb_batch) * f - g * h
    t3 = -(a - lamb_batch) * (t1 / t2) / g - (h / g)

    # Compute m, l, n
    m_batch = 1 / torch.sqrt(1 + torch.pow(t1 / t2, 2) + torch.pow(t3, 2))
    l_batch = (t1 / t2) * m_batch
    n_batch = t3 * m_batch

    return l_batch, m_batch, n_batch

'''
Safaee-Rad, 1992 (12), (27)-(33)

'''
def gen_lmn(lamb1, lamb2, lamb3):
    if lamb1 < lamb2:
        l = 0
        m_pos = np.sqrt((lamb2-lamb1)/(lamb2-lamb3))
        m_neg = -m_pos
        n = np.sqrt((lamb1-lamb3)/(lamb2-lamb3))
        return [l, l], [m_pos, m_neg], [n, n]
    elif lamb1 > lamb2:
        l_pos = np.sqrt((lamb1-lamb2)/(lamb1-lamb3))
        l_neg = -l_pos
        n = np.sqrt((lamb2-lamb3)/(lamb1-lamb3))
        m = 0
        return [l_pos, l_neg], [m, m], [n, n]
    elif lamb1 == lamb2:
        n = 1
        m = 0
        l = 0
        return [l,l], [m,m], [n,n]
    else:
        
        logging.warning("Failure to generate l,m,n. None's are returned")
        return None, None, None
    
def calT3(l, m ,n ):
    lm_sqrt = np.sqrt((l**2)+(m**2))
    T3 = np.array([-m/lm_sqrt, -(l*n)/lm_sqrt, l, 0,
                       l/lm_sqrt, -(m*n)/lm_sqrt, m, 0,
                       0, lm_sqrt, n, 0,
                       0, 0, 0, 1]).reshape(4,4)
    return T3


def calABCD(T3, lamb1, lamb2, lamb3):
    li, mi, ni = T3[0:3,0], T3[0:3,1], T3[0:3,2]
    lamb_array = np.array([lamb1, lamb2, lamb3])
    A = np.dot(np.power(li,2), lamb_array)
    B = np.sum(li*ni*lamb_array)
    C = np.sum(mi*ni*lamb_array)
    D = np.dot(np.power(ni,2), lamb_array)
    return A,B,C,D


def calXYZ_perfect(A,B,C,D, r):
    
    Z = (A*r)/np.sqrt((B**2)+(C**2)-A*D)
    X = (-B/A)*Z
    Y = (-C/A)*Z
    center = np.array([X,Y,Z,1]).reshape(4,1)
    return center
    
    
def check_parallel(v1, v2):
    a = np.dot(v1.T, v2)
    b = np.linalg.norm(v1) * np.linalg.norm(v2)
    radian = np.arccos(a/b).squeeze()
    return np.rad2deg(radian)

import torch

def gen_lmn_batch(lamb1_batch, lamb2_batch, lamb3_batch):
    """
    Vectorized version of gen_lmn for batches.
    Returns:
        l_batch: (batch_size, 2)
        m_batch: (batch_size, 2)
        n_batch: (batch_size, 2)
    """
    device = lamb1_batch.device
    batch_size = lamb1_batch.shape[0]
    dtype = lamb1_batch.dtype
    l_batch = torch.zeros((batch_size, 2), dtype=dtype, device=device)
    m_batch = torch.zeros((batch_size, 2), dtype=dtype, device=device)
    n_batch = torch.zeros((batch_size, 2), dtype=dtype, device=device)

    # Case 1: lamb1 < lamb2
    mask1 = lamb1_batch < lamb2_batch
    if mask1.any():  # Check if any element satisfies the condition
        m_pos = torch.sqrt((lamb2_batch[mask1] - lamb1_batch[mask1]) / (lamb2_batch[mask1] - lamb3_batch[mask1]))
        m_batch[mask1, 0] = m_pos
        m_batch[mask1, 1] = -m_pos
        n_batch[mask1, :] = torch.sqrt((lamb1_batch[mask1] - lamb3_batch[mask1]) / (lamb2_batch[mask1] - lamb3_batch[mask1])).unsqueeze(-1)

    # Case 2: lamb1 > lamb2
    mask2 = lamb1_batch > lamb2_batch
    if mask2.any():  # Check if any element satisfies the condition
        l_pos = torch.sqrt((lamb1_batch[mask2] - lamb2_batch[mask2]) / (lamb1_batch[mask2] - lamb3_batch[mask2]))
        l_batch[mask2, 0] = l_pos
        l_batch[mask2, 1] = -l_pos
        n_batch[mask2, :] = torch.sqrt((lamb2_batch[mask2] - lamb3_batch[mask2]) / (lamb1_batch[mask2] - lamb3_batch[mask2])).unsqueeze(-1)

    # Case 3: lamb1 == lamb2
    mask3 = lamb1_batch == lamb2_batch
    if mask3.any():  # Check if any element satisfies the condition
        n_batch[mask3, :] = 1.0

    return l_batch, m_batch, n_batch


def calT3_batch(l_batch, m_batch, n_batch):
    """
    Vectorized version of calT3 for batches.
    Returns:
        T3_batch: (batch_size, 4, 4)
    """
    device = l_batch.device
    batch_size = l_batch.shape[0]
    dtype = l_batch.dtype
    lm_sqrt = torch.sqrt(l_batch**2 + m_batch**2)
    T3_batch = torch.zeros((batch_size, 4, 4), dtype=dtype, device=device)

    T3_batch[:, 0, 0] = -m_batch / lm_sqrt
    T3_batch[:, 0, 1] = -(l_batch * n_batch) / lm_sqrt
    T3_batch[:, 0, 2] = l_batch
    T3_batch[:, 1, 0] = l_batch / lm_sqrt
    T3_batch[:, 1, 1] = -(m_batch * n_batch) / lm_sqrt
    T3_batch[:, 1, 2] = m_batch
    T3_batch[:, 2, 1] = lm_sqrt
    T3_batch[:, 2, 2] = n_batch
    T3_batch[:, 3, 3] = 1.0

    return T3_batch


def calABCD_batch(T3_batch, lamb1_batch, lamb2_batch, lamb3_batch):
    """
    Vectorized version of calABCD for batches.
    Returns:
        A_batch, B_batch, C_batch, D_batch: Each of shape (batch_size,)
    """
    li = T3_batch[:, 0:3, 0]
    mi = T3_batch[:, 0:3, 1]
    ni = T3_batch[:, 0:3, 2]
    lamb_array = torch.stack([lamb1_batch, lamb2_batch, lamb3_batch], dim=1)

    A_batch = torch.sum(li**2 * lamb_array, dim=1)
    B_batch = torch.sum(li * ni * lamb_array, dim=1)
    C_batch = torch.sum(mi * ni * lamb_array, dim=1)
    D_batch = torch.sum(ni**2 * lamb_array, dim=1)

    return A_batch, B_batch, C_batch, D_batch


def calXYZ_perfect_batch(A_batch, B_batch, C_batch, D_batch, r):
    """
    Vectorized version of calXYZ_perfect for batches.
    Returns:
        center_batch: (batch_size, 4, 1)
    """
    Z = (A_batch * r) / torch.sqrt(B_batch**2 + C_batch**2 - A_batch * D_batch)
    X = (-B_batch / A_batch) * Z
    Y = (-C_batch / A_batch) * Z
    center_batch = torch.stack([X, Y, Z, torch.ones_like(X)], dim=1).unsqueeze(-1)
    return center_batch


def convert_ell_to_general(xc,yc, w,h,radian):
    A = (w**2)*(np.sin(radian)**2) + (h**2) * (np.cos(radian)**2)
    B = 2 * ((h**2) - (w**2)) * np.sin(radian) * np.cos(radian)
    C = (w**2)*(np.cos(radian)**2) + (h**2)*(np.sin(radian)**2)
    D = -2*A*xc - B*yc
    E = -B*xc - 2*C*yc
    F = A*(xc**2) + B*xc*yc + C*(yc**2) - (w**2)*(h**2)
    return A,B,C,D,E,F

def convert_ell_to_general_batch(xc,yc, w,h,radian):
    A = (w**2)*(torch.sin(radian)**2) + (h**2) * (torch.cos(radian)**2)
    B = 2 * ((h**2) - (w**2)) * torch.sin(radian) * torch.cos(radian)
    C = (w**2)*(torch.cos(radian)**2) + (h**2)*(torch.sin(radian)**2)
    D = -2*A*xc - B*yc
    E = -B*xc - 2*C*yc
    F = A*(xc**2) + B*xc*yc + C*(yc**2) - (w**2)*(h**2)
    return A,B,C,D,E,F

def unprojectGazePositions(vertex, ell_co, radius = None):
    """
    This function generates (1)directions of unprojected pupil disk (gaze vector) 
    and (2) position of the pupil disk, with an assumed radius of the pupil disk

    Args:
        vectex (list or tuple): list with 3 elements of x, y, z coordinates of the camera with respect to the image frame
        ell_co (list or tuple): list of 6 coefficients of a generalised/expanded ellipse equations at the image frame
            A*(x**2) + B*x*y + C*(y**2) + D*x + E*y + F = 0 (from https://en.wikipedia.org/wiki/Ellipse#General_ellipse)
        
    Returns:
        Positive Norm of pupil disk from camera frame
        Negative Norm of pupil disk from camera frame
        Positive Norm of pupil disk from canonical frame
        negative Norm of pupil disk from canonical frame
    """
    
    # Coefficients of the general ellipse equation
    A,B,C,D,E,F = [x for x in ell_co]
    # Vertex (Point of the camera)
    alpha, beta, gamma = [x for x in vertex]
    
    # Ellipse parameter at image frame (z_c = +20) with respect to the camera frame
    a_prime = A
    h_prime = B/2
    b_prime = C
    g_prime = D/2
    f_prime = E/2
    d_prime = F
    # Coefficients of the Cone at the image frame
    a,b,c,d,f,g,h,u,v,w = gen_cone_co(alpha, beta, gamma, a_prime, h_prime, b_prime, g_prime, f_prime, d_prime)
    # Safaee-Rad, 1992 (10)
    lamb_co1 = 1
    lamb_co2 = -(a+b+c)
    lamb_co3 = (b*c + c*a + a*b - np.power(f,2) - np.power(g,2) - np.power(h,2))
    lamb_co4 = -(a*b*c + 2*f*g*h - a*np.power(f,2) - b*np.power(g,2) - c*np.power(h,2))
    lamb1, lamb2, lamb3 = np.roots([lamb_co1, lamb_co2, lamb_co3, lamb_co4])
    # generate Normal vector at the canonical frame
    
    l, m, n = gen_lmn(lamb1,lamb2,lamb3)
    norm_cano_pos = np.array([l[0],m[0],n[0],1]).reshape(4,1)
    norm_cano_neg = np.array([l[1],m[1],n[1],1]).reshape(4,1)
    
    # T1 Rotational Transformation to the camera fream
    l1, m1, n1 = gen_rotmat_co(lamb1, a,b,g,f,h)
    l2, m2, n2 = gen_rotmat_co(lamb2, a,b,g,f,h)
    l3, m3, n3 = gen_rotmat_co(lamb3, a,b,g,f,h)
    T1 = np.array([l1,l2,l3,0 ,m1, m2, m3,0, n1, n2, n3,0, 0,0,0,1]).reshape(4,4)
    li, mi, ni = T1[0,0:3], T1[1,0:3], T1[2,0:3]
    if np.cross(li,mi).dot(ni) < 0:
        li = -li
        mi = -mi
        ni = -ni
    T1[0,0:3], T1[1,0:3], T1[2,0:3] = li, mi, ni
    norm_cam_pos = np.dot(T1,  norm_cano_pos)
    norm_cam_neg = np.dot(T1, norm_cano_neg)

    # Calculating T2
    T2 = np.eye(4)
    T2[0:3,3] = -(u*li+v*mi+w*ni)/np.array([lamb1, lamb2, lamb3])
    # Calculating T3
    T3_pos = calT3(l[0], m[0], n[0])
    T3_neg = calT3(l[1], m[1], n[1])
    # calculate ABCD
    A_pos, B_pos, C_pos, D_pos = calABCD(T3_pos, lamb1, lamb2, lamb3)
    A_neg, B_neg, C_neg, D_neg = calABCD(T3_neg, lamb1, lamb2, lamb3)
    # Calculating T0
    T0 = np.eye(4)

    T0[2,3] = -gamma # -gamma = -(vertex[2]) = -(-focal_length) = + focal_length
    # Calculating center position with respect to the perfect frame
    center_pos = calXYZ_perfect(A_pos, B_pos, C_pos, D_pos, radius)
    center_neg = calXYZ_perfect(A_neg, B_neg, C_neg, D_neg, radius)
    # From perfect frame to camera frame
    true_center_pos = np.matmul(T0,np.matmul(T1,np.matmul(T2,np.matmul(T3_pos,center_pos))))
    if true_center_pos[2] <0:
        center_pos[0:3] = -center_pos[0:3]
        true_center_pos = np.matmul(T0,np.matmul(T1,np.matmul(T2,np.matmul(T3_pos,center_pos))))
    true_center_neg = np.matmul(T0,np.matmul(T1,np.matmul(T2,np.matmul(T3_neg,center_neg))))
    if true_center_neg[2] <0:
        center_neg[0:3] = -center_neg[0:3]
        true_center_neg = np.matmul(T0,np.matmul(T1,np.matmul(T2,np.matmul(T3_neg,center_neg))))

    return norm_cam_pos[0:3], norm_cam_neg[0:3], true_center_pos[0:3], true_center_neg[0:3]


def batched_roots(coefficients: torch.Tensor):
    """
    Compute and sort the roots of a batch of cubic polynomials using PyTorch.

    Args:
        coefficients (torch.Tensor): A tensor of shape (batch_size, 4) containing
                                     polynomial coefficients [a, b, c, d] for ax^3 + bx^2 + cx + d.

    Returns:
        torch.Tensor: A tensor of shape (batch_size, 3) containing sorted roots (largest to smallest).
    """
    batch_size = coefficients.shape[0]

    # Extract coefficients
    a, b, c, d = coefficients[:, 0], coefficients[:, 1], coefficients[:, 2], coefficients[:, 3]

    # Avoid division by zero if a == 0 (not a cubic polynomial)
    mask = a.abs() > 1e-7  # Ensure a is nonzero
    a_safe = torch.where(mask, a, torch.ones_like(a))  # Replace zero coefficients with 1 (avoiding NaN)

    # Normalize coefficients (convert to monic polynomial: x^3 + (b/a)x^2 + (c/a)x + (d/a))
    b, c, d = b / a_safe, c / a_safe, d / a_safe

    # Build batch companion matrices
    companion = torch.zeros((batch_size, 3, 3), device=coefficients.device, dtype=coefficients.dtype)
    companion[:, 0, 1] = 1
    companion[:, 1, 2] = 1
    companion[:, 2, 0] = -d
    companion[:, 2, 1] = -c
    companion[:, 2, 2] = -b

    # Compute eigenvalues (roots)
    roots = torch.linalg.eigvals(companion).real  # Keep only real part

    # Sort in descending order (biggest root first)
    roots_sorted, _ = torch.sort(roots, descending=True, dim=1)

    return roots_sorted  # Shape: (batch_size, 3)


def unprojectGazePositions_batch(vertex, ell_co, radius = None, valid_ix = None):
    """
    This function generates (1)directions of unprojected pupil disk (gaze vector) 
    and (2) position of the pupil disk, with an assumed radius of the pupil disk

    Args:
        vectex (list or tuple): list with 3 elements of x, y, z coordinates of the camera with respect to the image frame
        ell_co (list or tuple): list of 6 coefficients of a generalised/expanded ellipse equations at the image frame
            A*(x**2) + B*x*y + C*(y**2) + D*x + E*y + F = 0 (from https://en.wikipedia.org/wiki/Ellipse#General_ellipse)
        
    Returns:
        Positive Norm of pupil disk from camera frame
        Negative Norm of pupil disk from camera frame
        Positive Norm of pupil disk from canonical frame
        negative Norm of pupil disk from canonical frame
    """
    
    # Coefficients of the general ellipse equation
    A,B,C,D,E,F = ell_co
    # Vertex (Point of the camera)
    dtype = A.dtype
    alpha, beta, gamma = torch.tensor(vertex, dtype=dtype)
    
    # Ellipse parameter at image frame (z_c = +20) with respect to the camera frame
    a_prime = A
    h_prime = B/2
    b_prime = C
    g_prime = D/2
    f_prime = E/2
    d_prime = F
    # Coefficients of the Cone at the image frame
    a,b,c,d,f,g,h,u,v,w = gen_cone_co_torch(alpha, beta, gamma, a_prime, h_prime, b_prime, g_prime, f_prime, d_prime)
    # Safaee-Rad, 1992 (10)
    batch_size = a.shape[0]
    device = a.device   
    lamb_co1 = torch.ones(batch_size, dtype=dtype, device=device)
    lamb_co2 = -(a+b+c)
    lamb_co3 = (b*c + c*a + a*b - torch.pow(f,2) - torch.pow(g,2) - torch.pow(h,2))
    lamb_co4 = -(a*b*c + 2*f*g*h - a*torch.pow(f,2) - b*torch.pow(g,2) - c*torch.pow(h,2))

    roots = torch.full((batch_size, 3), torch.nan, dtype=dtype, device= device)
    coefficients = (torch.stack([lamb_co1[valid_ix], lamb_co2[valid_ix], lamb_co3[valid_ix], lamb_co4[valid_ix]], dim=1))
    # Compute roots for each row of coefficients and convert back to PyTorch tensors
    # Note: np.apply_along_axis won't accelerate computation speed than using for loop
    # roots[valid_ix] = torch.tensor(np.apply_along_axis(np.roots, 1, coefficients.cpu().numpy()), dtype=dtype, device=device)
    roots[valid_ix] = batched_roots(coefficients)
    lamb1_batch, lamb2_batch, lamb3_batch = roots[:, 0], roots[:, 1], roots[:, 2]

    # # Number of polynomials
    # num_polynomials, degree_plus_one = coefficients.shape
    # degree = degree_plus_one - 1
    # # Normalize coefficients so that the leading coefficient is 1 for each polynomial
    # normalized_coefficients = coefficients / coefficients[:, 0:1]
    # # Create companion matrices for all polynomials
    # companion_matrices = torch.zeros((num_polynomials, degree, degree), dtype=dtype)
    # companion_matrices[:, 1:, :-1] = torch.eye(degree - 1).expand(num_polynomials, -1, -1)  # Subdiagonal 1s
    # companion_matrices[:, :, -1] = -normalized_coefficients[:, 1:]  # Last column: normalized coefficients
    # # Compute eigenvalues (roots) of the companion matrices
    # roots[valid_ix,:] = torch.linalg.eigvals(companion_matrices)
    # lamb1_batch, lamb2_batch, lamb3_batch = roots[:,0].real, roots[:,1].real, roots[:,2].real

    # Generate l, m, n in batch
    l_batch, m_batch, n_batch = gen_lmn_batch(lamb1_batch, lamb2_batch, lamb3_batch)

    # Generate norm_cano_pos and norm_cano_neg in batch
    norm_cano_pos = torch.stack([l_batch[:, 0], m_batch[:, 0], n_batch[:, 0], torch.ones(batch_size, device= device)], dim=1).unsqueeze(-1)
    norm_cano_neg = torch.stack([l_batch[:, 1], m_batch[:, 1], n_batch[:, 1], torch.ones(batch_size, device = device)], dim=1).unsqueeze(-1)

    # Generate T1 in batch
    l1, m1, n1 = gen_rotmat_co_batch(lamb1_batch, a, b, g, f, h)
    l2, m2, n2 = gen_rotmat_co_batch(lamb2_batch, a, b, g, f, h)
    l3, m3, n3 = gen_rotmat_co_batch(lamb3_batch, a, b, g, f, h)
    T1 = torch.zeros((batch_size, 4, 4), dtype=dtype, device = device)
    T1[:, 0, 0:3] = torch.stack([l1, l2, l3], dim=1)
    T1[:, 1, 0:3] = torch.stack([m1, m2, m3], dim=1)
    T1[:, 2, 0:3] = torch.stack([n1, n2, n3], dim=1)
    T1[:, 3, 3] = 1.0

    # Extract li, mi, ni from T1
    li = T1[:, 0, 0:3]  # Shape: (batch_size, 3)
    mi = T1[:, 1, 0:3]  # Shape: (batch_size, 3)
    ni = T1[:, 2, 0:3]  # Shape: (batch_size, 3)

    # Compute cross product (li x mi) for each batch element
    cross_li_mi = torch.cross(li, mi, dim=1)  # Shape: (batch_size, 3)
    dot_product = torch.sum(cross_li_mi * ni, dim=1)  # Shape: (batch_size,)
    mask = dot_product < 0  # Shape: (batch_size,)
    # Flip the sign of li, mi, ni for elements where the mask is True
    li[mask] = -li[mask]
    mi[mask] = -mi[mask]
    ni[mask] = -ni[mask]
    T1[:,0,0:3], T1[:,1,0:3], T1[:,2,0:3] = li, mi, ni
    # Compute norm_cam_pos and norm_cam_neg in batch
    norm_cam_pos = torch.matmul(T1, norm_cano_pos)
    norm_cam_neg = torch.matmul(T1, norm_cano_neg)

    # Compute T2 in batch
    T2 = torch.eye(4, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    # Expand the shape of (u * li[:, 0] + v * mi[:, 0] + w * ni[:, 0]) to [32, 1]
    numerator = (u.unsqueeze(1) * li + v.unsqueeze(1) * mi + w.unsqueeze(1) * ni) 
    # Divide by torch.stack([lamb1_batch, lamb2_batch, lamb3_batch], dim=1) (Shape: [32, 3])
    T2[:, 0:3, 3] = -numerator / torch.stack([lamb1_batch, lamb2_batch, lamb3_batch], dim=1)

    # Compute T3_pos and T3_neg in batch
    T3_pos = calT3_batch(l_batch[:, 0], m_batch[:, 0], n_batch[:, 0])
    T3_neg = calT3_batch(l_batch[:, 1], m_batch[:, 1], n_batch[:, 1])

    # Compute A, B, C, D in batch
    A_pos, B_pos, C_pos, D_pos = calABCD_batch(T3_pos, lamb1_batch, lamb2_batch, lamb3_batch)
    A_neg, B_neg, C_neg, D_neg = calABCD_batch(T3_neg, lamb1_batch, lamb2_batch, lamb3_batch)

    # Compute center_pos and center_neg in batch
    center_pos = calXYZ_perfect_batch(A_pos, B_pos, C_pos, D_pos, radius)
    center_neg = calXYZ_perfect_batch(A_neg, B_neg, C_neg, D_neg, radius)

    # Compute true_center_pos and true_center_neg in batch
    T0 = torch.eye(4, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1).to(device)
    T0[:, 2, 3] = -gamma

    true_center_pos = torch.matmul(T0, torch.matmul(T1, torch.matmul(T2, torch.matmul(T3_pos, center_pos))))
    true_center_neg = torch.matmul(T0, torch.matmul(T1, torch.matmul(T2, torch.matmul(T3_neg, center_neg))))

    # Handle negative z-values
    mask_pos = true_center_pos[:, 2, 0] < 0
    center_pos[mask_pos, 0:3, 0] = -center_pos[mask_pos, 0:3, 0]
    true_center_pos[mask_pos] = torch.matmul(T0[mask_pos], torch.matmul(T1[mask_pos], torch.matmul(T2[mask_pos], torch.matmul(T3_pos[mask_pos], center_pos[mask_pos]))))

    mask_neg = true_center_neg[:, 2, 0] < 0
    center_neg[mask_neg, 0:3, 0] = -center_neg[mask_neg, 0:3, 0]
    true_center_neg[mask_neg] = torch.matmul(T0[mask_neg], torch.matmul(T1[mask_neg], torch.matmul(T2[mask_neg], torch.matmul(T3_neg[mask_neg], center_neg[mask_neg]))))

    return norm_cam_pos[:, 0:3, 0], norm_cam_neg[:, 0:3, 0], true_center_pos[:, 0:3, 0], true_center_neg[:, 0:3, 0]


def reproject(vec_3d, focal_length, batch_mode= False):
    # vec_3d = (3,1) numpy array: Coordinates of the 3D unprojected object in CAMERA frame
    # vec_3d can also be (3,), but not (1,3)
    focal_length = torch.tensor(focal_length)
    if batch_mode == False:
        vec_2d = (focal_length*vec_3d[0:2])/vec_3d[2]
    else:
        focal_length = focal_length
        # converting vec_3d ~ (m,3) to vec_2d~(m,2)
        vec_2d = (focal_length*(vec_3d[:,0:2]))/vec_3d[:,[2]]
    return vec_2d
    
def reverse_reproject(vec_2d, z, focal_length):
    # Scale the x,y in a reverse manner of reproject() function,
    # when you unproject the reprojected coordinate.
    vec_2d_scaled = (vec_2d*z)/focal_length
    return vec_2d_scaled

# Illustration of the example from Safaee-Rad's paper
if __name__ == "__main__":
    pass
    
    