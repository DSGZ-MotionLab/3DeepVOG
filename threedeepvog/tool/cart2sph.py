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
def cart2sph_batch_PL(N):
    N = N / torch.linalg.norm(N, dim=0, keepdims=True)  # Normalize each vector
    phi = torch.arctan2(N[2], N[0]) + torch.pi / 2
    theta = torch.arccos(N[1]) - torch.pi/2
    return torch.rad2deg(phi), torch.rad2deg(theta)

def cart2sph_batch(N):
    N = N / torch.linalg.norm(N, dim=0, keepdims=True)  # Normalize each vector
    phi = -(torch.arctan2(N[2], N[0]) + torch.pi / 2)    
    theta = -(torch.arctan2(N[2], N[1]) - torch.pi / 2)
    # theta = -(np.arccos(N[1]) - np.pi/2)
    return torch.rad2deg(phi), torch.rad2deg(theta)
