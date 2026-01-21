import numpy as np
import torch

class NoIntersectionError(Exception):
    pass

# vector (m,n) , m = number of examples, n = dimensionality
# a = coordinates of the vector
# n = orientation of the vector
def intersect(a, n):
    # default normalisation of vectors n
    n = n/np.linalg.norm(n, axis = 1, keepdims=True)
    num_lines = a.shape[0]
    dim = a.shape[1]
    I = np.eye(dim)
    R_sum = 0
    q_sum = 0
    for i in range(num_lines):    
        R = I - np.matmul(n[i].reshape(dim,1), n[i].reshape(1,dim))

        q = np.matmul(R,a[i].reshape(dim,1))
        q_sum = q_sum + q
        R_sum = R_sum + R
    p = np.matmul(np.linalg.inv(R_sum),q_sum)
    return p



def calc_distance(a,n, p):
    num_lines = a.shape[0]
    dim = a.shape[1]
    I = np.eye(dim)
    D_sum = 0
    D_sum_dist = 0
    for i in range(num_lines):
        D_1 = (a[i].reshape(dim,1) - p.reshape(dim,1)).T
        D_2 = I - np.matmul(n[i].reshape(dim,1), n[i].reshape(1,dim))
        D_3 = D_1.T
        # D = np.matmul(np.matmul(D_1,D_2),D_3)
        D_dist = np.linalg.norm(np.matmul(D_1,D_2))
        # D_sum = D_sum + D
        D_sum_dist = D_sum_dist + D_dist
    D_sum = D_sum/num_lines
    D_sum_dist = D_sum_dist/num_lines
    return D_sum_dist


def fit_ransac(a,n, max_iters = 2000, samples_to_fit = 20, min_distance = 2000):
    num_lines = a.shape[0]
    best_model = None
    best_distance = min_distance
    for i in range(max_iters):
        # print("\rRANSAC: Currently {0}".format(i), flush=True)
        sampling_index = np.random.choice(num_lines, size = samples_to_fit, replace=False)
        a_sampled = a[sampling_index,:]
        n_sampled = n[sampling_index,:]
        model_sampled = intersect(a_sampled, n_sampled)
        sampled_distance = calc_distance( a,n, model_sampled)
        # print(sampled_distance)
        if sampled_distance > min_distance:
            continue
        else:
            if sampled_distance < best_distance:
                best_model = model_sampled
                best_distance = sampled_distance
    # if best_model is None:
    #     best_model = model_sampled
    return best_model, best_distance


def calc_distance_batch(a_sampled_batch, n_sampled_batch, p_batch, device = 'cpu'):
    num_batch, num_sample, dim = a_sampled_batch.shape
    # Identity matrix batch 
    D1_sampled_batch = (a_sampled_batch.unsqueeze(-1) - p_batch.unsqueeze(1)).transpose(-1,-2)
    
    I_sampled_batch = torch.eye(dim, device=device).expand(num_batch, num_sample, dim, dim)
    outer_products = torch.einsum('bsi,bsj->bsij', n_sampled_batch, n_sampled_batch)
    D2_sampled_batch = I_sampled_batch - outer_products  #batch x sample x dim x dim
    D_sampled_batch = torch.linalg.norm(torch.matmul(D1_sampled_batch, D2_sampled_batch), dim=(-1,-2))
    median_all = torch.mean(D_sampled_batch)
    Dmean_batch = torch.mean(D_sampled_batch, dim=1)
    # num_inlier = torch.sum(D_sampled_batch > torch.mean(D_sampled_batch), dim=1)
    return Dmean_batch

def calc_distance_batch_v2(a_sampled_batch, n_sampled_batch, p_batch, device = 'cpu'):
    num_batch, num_sample, dim = a_sampled_batch.shape
    # Identity matrix batch 
    D1_sampled_batch = (a_sampled_batch.unsqueeze(-1) - p_batch.unsqueeze(1)).transpose(-1,-2)
    
    I_sampled_batch = torch.eye(dim, device=device).expand(num_batch, num_sample, dim, dim)
    outer_products = torch.einsum('bsi,bsj->bsij', n_sampled_batch, n_sampled_batch)
    D2_sampled_batch = I_sampled_batch - outer_products  #batch x sample x dim x dim
    D_sampled_batch = torch.linalg.norm(torch.matmul(D1_sampled_batch, D2_sampled_batch), dim=(-1,-2))
    
    # Dmean_batch = torch.mean(D_sampled_batch, dim=1)
    return D_sampled_batch

def calc_distance_batch_v3(a_sampled_batch, n_sampled_batch, p_batch, device = 'cpu'):
    num_batch, num_sample, dim = a_sampled_batch.shape
    # Identity matrix batch 
    D1_sampled_batch = (a_sampled_batch.unsqueeze(-1) - p_batch.unsqueeze(1)).transpose(-1,-2)
    
    I_sampled_batch = torch.eye(dim, device=device).expand(num_batch, num_sample, dim, dim)
    outer_products = torch.einsum('bsi,bsj->bsij', n_sampled_batch, n_sampled_batch)
    D2_sampled_batch = I_sampled_batch - outer_products  #batch x sample x dim x dim
    D_sampled_batch = torch.linalg.norm(torch.matmul(D1_sampled_batch, D2_sampled_batch), dim=(-1,-2))
    # median_all = torch.mean(D_sampled_batch)
    # Dmean_batch = torch.mean(D_sampled_batch, dim=1)
    num_inlier = torch.sum(D_sampled_batch < torch.median(D_sampled_batch), dim=1)  #smallest 50% distances
    return num_inlier

def intersect_batch(a_sampled_batch, n_sampled_batch, device = 'cpu'):
    '''
    a_sampled_batch: torch.Tensor of shape [batch, sample, dim]
    n_sampled_batch: torch.Tensor of shape [batch, sample, dim]
    '''
    n_sampled_batch = n_sampled_batch/torch.linalg.norm(n_sampled_batch, axis = -1, keepdims=True)
    num_batch, num_sample, dim = a_sampled_batch.shape
    device = n_sampled_batch.device
    # Identity matrix batch 
    I_sampled_batch = torch.eye(dim, device=device).expand(num_batch, num_sample, dim, dim)
    # outer_products = n_sampled_batch.unsqueeze(-1) * n_sampled_batch.unsqueeze(-2)  #batch x sample x dim x dim
    outer_products = torch.einsum('bsi,bsj->bsij', n_sampled_batch, n_sampled_batch)
    R_sampled_batch = I_sampled_batch - outer_products  #batch x sample x dim x dim
    # q = torch.einsum('bsij,bsi->bsi', R, a_sampled_batch)  #batch x sample x dim x 1
    q_sampled_batch = torch.matmul(R_sampled_batch, a_sampled_batch.unsqueeze(-1))
    R_sum_batch = torch.sum(R_sampled_batch, dim=1)   #batch x dim x dim
    q_sum_batch = torch.sum(q_sampled_batch, dim=1)   #batch x dim x 1
    # Compute the batched inverse of R_sum
    p_batch = torch.matmul(torch.linalg.inv(R_sum_batch), q_sum_batch)  # batch x 2 x 1
    return p_batch

def intersect_batch_v2(a_sampled_batch, n_sampled_batch, weights, beta = 0, prior_c = None, device = 'cpu'):
    '''
    a_sampled_batch: torch.Tensor of shape [batch, sample, dim]
    n_sampled_batch: torch.Tensor of shape [batch, sample, dim]
    '''
    n_sampled_batch = n_sampled_batch/torch.linalg.norm(n_sampled_batch, axis = -1, keepdims=True)
    num_batch, num_sample, dim = a_sampled_batch.shape
    device = n_sampled_batch.device
    # Identity matrix batch 
    I_sampled_batch = torch.eye(dim, device=device).expand(num_batch, num_sample, dim, dim)
    # outer_products = n_sampled_batch.unsqueeze(-1) * n_sampled_batch.unsqueeze(-2)  #batch x sample x dim x dim
    outer_products = torch.einsum('bsi,bsj->bsij', n_sampled_batch, n_sampled_batch)
    R_sampled_batch = I_sampled_batch - outer_products  #batch x sample x dim x dim
    # q = torch.einsum('bsij,bsi->bsi', R, a_sampled_batch)  #batch x sample x dim x 1
    q_sampled_batch = torch.matmul(R_sampled_batch, a_sampled_batch.unsqueeze(-1))
    weights = weights.view(weights.shape[0], weights.shape[1], 1, 1)
    if prior_c is None:
        prior_c = 0
    R_sum_batch = torch.sum(R_sampled_batch * weights + I_sampled_batch * beta, dim=1)   #batch x dim x dim
    q_sum_batch = torch.sum(q_sampled_batch * weights + beta*prior_c, dim=1)   #batch x dim x 1
    # R_sum_batch = torch.sum(R_sampled_batch, dim=1)   #batch x dim x dim
    # q_sum_batch = torch.sum(q_sampled_batch, dim=1)   #batch x dim x 1
    # Compute the batched inverse of R_sum
    p_batch = torch.matmul(torch.linalg.inv(R_sum_batch), q_sum_batch)
    return p_batch

def fit_eyeball_radius_batch(a_sampled_batch, eye_centre, n_sampled_batch):
    n_sampled_batch = n_sampled_batch/torch.linalg.norm(n_sampled_batch, axis = -1, keepdims=True)
    num_batch, num_sample, dim = a_sampled_batch.shape
    device = n_sampled_batch.device
    # Identity matrix batch 
    I_sampled_batch = torch.eye(dim, device=device).expand(num_batch, num_sample, dim, dim)
    # outer_products = n_sampled_batch.unsqueeze(-1) * n_sampled_batch.unsqueeze(-2)  #batch x sample x dim x dim
    outer_products = torch.einsum('bsi,bsj->bsij', n_sampled_batch, n_sampled_batch)
    R_sampled_batch = I_sampled_batch - outer_products  #batch x sample x dim x dim
    # q = torch.einsum('bsij,bsi->bsi', R, a_sampled_batch)  #batch x sample x dim x 1
    q_sampled_batch = torch.matmul(R_sampled_batch, a_sampled_batch.unsqueeze(-1))
    R_sum_batch = torch.sum(R_sampled_batch, dim=1)   #batch x dim x dim
    q_sum_batch = torch.sum(q_sampled_batch, dim=1)   #batch x dim x 1
    # Compute the batched inverse of R_sum
    p_batch = torch.matmul(torch.linalg.inv(R_sum_batch), q_sum_batch)  # batch x 2 x 1
    radius_batch = torch.linalg.norm(p_batch - eye_centre.unsqueeze(0), dim=1)  # Shape [668, 3]
    radius_counter = radius_batch.shape[0]
    aver_eye_radius = torch.median(radius_batch.squeeze()).item()
    return eye_centre, aver_eye_radius, radius_counter


def fit_ransac_batch(a,n, max_iters = 2000, samples_to_fit = 20, min_distance = 2000, device = 'cpu'):
    '''
    GPU accelerated version of the RANSAC algorithm
    '''
    num_lines = a.shape[0]
    device = device
    best_model = None
    best_distance = None
    # sampling_index_batch = np.random.choice(num_lines, size=(max_iters, samples_to_fit))
    sampling_index_batch = (
        torch.arange(num_lines, device='cpu', dtype=torch.float32)  # Convert to float
        .repeat(max_iters, 1)  # Repeat for batch processing
        .multinomial(samples_to_fit, replacement=False)  # Sample without replacement
    )
    a_sampled_batch = a[sampling_index_batch, :]
    n_sampled_batch = n[sampling_index_batch, :]
    if a_sampled_batch.dtype == np.float32:
        n_sampled_batch = torch.from_numpy(n_sampled_batch).float().to(device)
        a_sampled_batch = torch.from_numpy(a_sampled_batch).float().to(device)
    model_sampled_batch = intersect_batch(a_sampled_batch, n_sampled_batch, device)
    # Dmean_batch = calc_distance_batch(a_sampled_batch, n_sampled_batch, model_sampled_batch, device)
    num_inlier = calc_distance_batch(a_sampled_batch, n_sampled_batch, model_sampled_batch, device)
    best_batch = torch.argmax(num_inlier).item()
    best_model = model_sampled_batch[best_batch].cpu().numpy()
    best_distance = num_inlier[best_batch].item()
    # best_batch = torch.argmin(Dmean_batch).item()
    # best_model = model_sampled_batch[best_batch].cpu().numpy()
    # best_distance = Dmean_batch[best_batch].item()
    return best_model, best_distance


def fit_ransac_batch_v2(a,n, batch_size = 2000, sample_size = 20):
    '''
    GPU accelerated version of the RANSAC algorithm
    '''
    num_lines = a.shape[0]
    device = a.device
    best_model = None
    dtype = a.dtype
    # sampling_index_batch = np.random.choice(num_lines, size=(max_iters, samples_to_fit))
    # Ensures unique indices per row
    sampling_index_batch = (
    torch.arange(num_lines, device=device, dtype=dtype)  # Convert to float
    .repeat(batch_size, 1)  # Repeat for batch processing
    .multinomial(sample_size, replacement=False)  # Sample without replacement
    )
    a_sampled_batch = a[sampling_index_batch, :]
    n_sampled_batch = n[sampling_index_batch, :]
    model_sampled_batch = intersect_batch(a_sampled_batch, n_sampled_batch, device)
    Dmean_batch = calc_distance_batch(a_sampled_batch, n_sampled_batch, model_sampled_batch, device)
    best_batch = torch.argmin(Dmean_batch).item()
    best_model = model_sampled_batch[best_batch]
    best_distance = Dmean_batch[best_batch].item()
    return best_model, best_distance

def fit_ransac_batch_v3(a,n, batch_size = 2000, sample_size = 20):
    min_batch_size = 10000   # max safe batch size for the GPU 
    num_iter = batch_size//min_batch_size
    num_lines = a.shape[0]
    device = a.device
    best_models = []
    best_distances = []
    dtype = a.dtype
    sample_pool = torch.arange(num_lines, device=device, dtype=dtype)
    for gen in range(num_iter):   #tqdm(
        sampled_ix_batch = (
        sample_pool  # Convert to float
        .repeat(min_batch_size, 1)  # Repeat for batch processing
        .multinomial(sample_size, replacement=False))
        
        a_sampled_batch = a[sampled_ix_batch, :]
        n_sampled_batch = n[sampled_ix_batch, :]
        model_sampled_batch = intersect_batch(a_sampled_batch, n_sampled_batch, device)
        fitness = calc_distance_batch(a_sampled_batch, n_sampled_batch, model_sampled_batch, device)
        best_batch = torch.argmin(fitness).item()
        best_models.append(model_sampled_batch[best_batch])
        best_distances.append(fitness[best_batch].item())
    best_ix = np.argmin(best_distances)
    best_model = best_models[best_ix]
    best_distance = best_distances[best_ix]
    return best_model, best_distance

def fit_ransac_batch_v4(a,n):
    # num_iter = batch_size//min_batch_size
    num_lines = a.shape[0]
    min_num_line = int(num_lines * 0.9)
    reduced_sample_size = int(num_lines * 0.01)
    device, dtype = a.device, a.dtype
    a_sampled_batch, n_sampled_batch = a.unsqueeze(0), n.unsqueeze(0)
    best_res_ratio = 1000000
    while min_num_line < a_sampled_batch.shape[1]:
        model_sampled_batch = intersect_batch(a_sampled_batch, n_sampled_batch, device = device)
        res_all_line = calc_distance_batch_v2(a_sampled_batch, n_sampled_batch, model_sampled_batch, device)
        # current_res_mean = torch.mean(res_all_line)
        values, indeces = torch.sort(res_all_line.squeeze())
        good_ix = indeces[:-reduced_sample_size]
        a_sampled_batch = a_sampled_batch[:,good_ix,:]
        n_sampled_batch = n_sampled_batch[:,good_ix,:]

        good_res_all = values[:-reduced_sample_size]
        good_res_mean, max_res, min_res = torch.mean(good_res_all), torch.max(good_res_all), torch.min(good_res_all)
        current_res_ratio = max_res/min_res
        # if selected_res_mean < current_residual:
        if current_res_ratio < best_res_ratio:
            best_ix = good_ix
            best_res_ratio = current_res_ratio
            best_model = intersect_batch(a_sampled_batch, n_sampled_batch, device = device).squeeze(0)
            best_distance = good_res_mean
        # else:
        #     selected_res_mean = current_residual
            # break

    return best_model, best_distance

def fit_ransac_batch_v5(a,n, weights = None, prior_weight=None, batch_size = 2000, sample_size = 20):
    # num_iter = batch_size//min_batch_size
    num_lines = a.shape[0]
    # min_num_line = int(num_lines * 0.80)
    # reduced_sample_size = int(num_lines * 0.01)
    device, dtype = a.device, a.dtype
    a_sampled_batch, n_sampled_batch, w_sampled_batch = a.unsqueeze(0), n.unsqueeze(0), weights.unsqueeze(0)
    prior_c = intersect_batch_v2(a_sampled_batch, n_sampled_batch, 
                                            weights = w_sampled_batch,
                                        device = device)

    min_batch_size = 10000   # max safe batch size for the GPU 
    num_iter = batch_size//min_batch_size
    best_models = []
    best_distances = []
    dtype = a.dtype
    sample_pool = torch.arange(num_lines, device=device, dtype=dtype)
    for gen in range(num_iter):   #tqdm(
        sampled_ix_batch = (
        sample_pool  # Convert to float
        .repeat(min_batch_size, 1)  # Repeat for batch processing
        .multinomial(sample_size, replacement=False))
        
        a_sampled_batch = a[sampled_ix_batch, :]
        n_sampled_batch = n[sampled_ix_batch, :]
        weight_sampled_batch = weights[sampled_ix_batch]
        model_sampled_batch = intersect_batch_v2(a_sampled_batch, n_sampled_batch, 
                                                 weights = weight_sampled_batch, 
                                                beta = prior_weight, prior_c = prior_c,
                                                device = device)

        fitness = calc_distance_batch(a_sampled_batch, n_sampled_batch, model_sampled_batch, device)
        best_batch = torch.argmin(fitness).item()
        best_models.append(model_sampled_batch[best_batch])
        best_distances.append(fitness[best_batch].item())
    best_ix = np.argmin(best_distances)
    best_model = best_models[best_ix]
    best_distance = best_distances[best_ix]
    return best_model, best_distance


def fit_ransac_batch_v6(a,n, batch_size = 2000, sample_size = 20):
    '''
    GPU accelerated version of the RANSAC algorithm
    '''
    num_lines = a.shape[0]
    device = a.device
    best_model = None
    dtype = a.dtype
    # sampling_index_batch = np.random.choice(num_lines, size=(max_iters, samples_to_fit))
    # Ensures unique indices per row
    sampling_index_batch = (
    torch.arange(num_lines, device=device, dtype=dtype)  # Convert to float
    .repeat(batch_size, 1)  # Repeat for batch processing
    .multinomial(sample_size, replacement=False)  # Sample without replacement
    )
    a_sampled_batch = a[sampling_index_batch, :]
    n_sampled_batch = n[sampling_index_batch, :]
    model_sampled_batch = intersect_batch(a_sampled_batch, n_sampled_batch, device)
    # num_inlier = calc_distance_batch_v3(a_sampled_batch, n_sampled_batch, model_sampled_batch, device)
        # Evaluate all models on all data
    all_a = a.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, N, 3]
    all_n = n.unsqueeze(0).expand(batch_size, -1, -1)
    num_inliers = calc_distance_batch_v3(all_a, all_n, model_sampled_batch, device)  # [batch_size]

    best_batch = torch.argmax(num_inliers).item()
    best_model = model_sampled_batch[best_batch]
    # best_distance = num_inlier[best_batch].item()
    return best_model


def mutate_batch(offspring, num_lines, mutation_rate=0.01):
    # Ensure offspring is a floating-point tensor
    if not offspring.is_floating_point():
        offspring = offspring.float()
    # Generate random values for mutation decisions
    random_values = torch.rand_like(offspring)
    mutation_mask = random_values < mutation_rate
    new_genes = torch.randint(0, num_lines, offspring.shape, device=offspring.device)
    offspring[mutation_mask] = new_genes[mutation_mask].float()
    return offspring

def crossover_batch(parents, parent_pairs):
    num_offspring = parent_pairs.shape[0]
    individual_length = parents.shape[1]
    # Initialize offspring tensor
    offspring = torch.empty((num_offspring, individual_length), dtype=parents.dtype, device=parents.device)
    # Random mask to decide gene selection from parents
    mask = torch.randint(0, 2, (num_offspring, individual_length), dtype=torch.bool, device=parents.device)
    # Gather parent genes based on the mask
    parent1_genes = parents[parent_pairs[:, 0]]
    parent2_genes = parents[parent_pairs[:, 1]]
    offspring[mask] = parent1_genes[mask]
    offspring[~mask] = parent2_genes[~mask]
    return offspring
    
def select_parents(population, fitness, num_parents):
    _, indices = torch.topk(fitness, num_parents)
    return population[indices]

def fit_GA_batch(a,n, batch_size = 2000, sample_size = 20, num_gen=50, num_parents = 20, mutation_rate = 0.01):
    '''
    GPU accelerated version of the GA algorithm
    '''
    num_lines = a.shape[0]
    device = a.device
    best_model = None
    dtype = a.dtype
    # sampling_index_batch = np.random.choice(num_lines, size=(max_iters, samples_to_fit))
    # Ensures unique indices per row
    sample_pool = torch.arange(num_lines, device=device, dtype=dtype)

    for gen in range(num_gen):   #tqdm(
        sampled_ix_batch = (
        sample_pool  # Convert to float
        .repeat(batch_size, 1)  # Repeat for batch processing
        .multinomial(sample_size, replacement=False)  # Sample without replacement
        )
        a_sampled_batch = a[sampled_ix_batch, :]
        n_sampled_batch = n[sampled_ix_batch, :]
        model_sampled_batch = intersect_batch(a_sampled_batch, n_sampled_batch, device)
        fitness = calc_distance_batch(a_sampled_batch, n_sampled_batch, model_sampled_batch, device)
        if gen == num_gen-1:
            best_batch = torch.argmin(fitness).item()
            best_model = model_sampled_batch[best_batch]
            best_distance = fitness[best_batch].item()
            break
        parents = select_parents(sampled_ix_batch, fitness, num_parents)  #to 20 were selected out of 2000 (max_iters)
        sample_pool = torch.unique(parents).to(dtype)
        if sample_pool.shape[0] <= sample_size:
            break
        # Select parent pairs
        # parent_indices = torch.randint(0, num_parents, (max_iters, 2))
        # offspring = crossover_batch(parents, parent_indices)   # Perform crossover to generate offspring
        # sampled_ix_batch = mutate_batch(offspring, num_lines, mutation_rate) # Apply mutation to offspring
        # sampled_ix_batch = sampled_ix_batch.int()
    return best_model, best_distance

    
def line_sphere_intersect(c, r, o, l):
    # c = numpy array (3,1). Centre of the eyeball
    # r = scaler. Radius of the eyeball
    # o = numpy array (3,1). Origin of the line
    # l = numpy array (3,1). Directional unit vector of the line
    # return [d1, d2] : auxilary variables of the parametrised line x = o + dl
    # the closer one to the camera is chosen
    l = l/np.linalg.norm(l)
    delta = np.square(np.dot(l.T,(o-c))) - np.dot((o-c).T,(o-c)) + np.square(r)
    if delta < 0:
        raise NoIntersectionError
    else:
        d1 = -np.dot(l.T,(o-c)) + np.sqrt(delta)
        d2 = -np.dot(l.T,(o-c)) - np.sqrt(delta)
    return [d1,d2]
    
def line_sphere_intersect_batch(c, r, o, l):
    # c = numpy array (3,1). Centre of the eyeball
    # r = scaler. Radius of the eyeball
    # o = numpy array (3,1). Origin of the line
    # l = numpy array (3,1). Directional unit vector of the line
    # return [d1, d2] : auxilary variables of the parametrised line x = o + dl
    # the closer one to the camera is chosen
    l = l/torch.linalg.norm(l, axis=1).reshape(-1,1)
    delta = torch.square(torch.matmul(l,(o-c))) - torch.matmul((o-c).T,(o-c)) + torch.square(r)
    IntersectionError_ix = (delta < 0).squeeze()
    d1 = -torch.matmul(l,(o-c)) + torch.sqrt(delta)
    d2 = -torch.matmul(l,(o-c)) - torch.sqrt(delta)
    return d1, d2, IntersectionError_ix

# def line_sphere_intersect_batch(c, r, o, l):
#     """
#     Args:
#         c: (B, 3) torch.Tensor — sphere centers (batch)
#         r: float — sphere radius (same for all)
#         o: (3,) torch.Tensor — ray origin (same for all)
#         l: (B, N, 3) torch.Tensor — ray direction vectors per batch element

#     Returns:
#         d1, d2: (B, N) torch.Tensor — two intersection distances per ray
#         invalid_mask: (B, N) torch.BoolTensor — True where no intersection
#     """
#     B, N, _ = l.shape
#     c = c.view(B, 1, 3)        # (B, 1, 3)
#     o = o.view(1, 1, 3)        # (1, 1, 3)
#     l = l / torch.norm(l, dim=2, keepdim=True)  # normalize (B, N, 3)

#     oc = o - c                 # (B, 1, 3)
#     oc = oc.expand(-1, N, -1)  # (B, N, 3)

#     b = torch.sum(l * oc, dim=2)              # (B, N)
#     oc_sq = torch.sum(oc * oc, dim=2)         # (B, N)
#     delta = b**2 - oc_sq + r**2               # (B, N)

#     invalid_batch = torch.any((delta < 0) | (torch.isnan(delta)), dim=1)  #B
#     sqrt_delta = torch.sqrt(torch.clamp(delta, min=0.0))  # Avoid NaNs
#     d1 = -b + sqrt_delta  # (B, N)
#     d2 = -b - sqrt_delta  # (B, N)
#     return d1, d2, invalid_batch
#%%
if __name__ == "__main__":
    
    pass

    