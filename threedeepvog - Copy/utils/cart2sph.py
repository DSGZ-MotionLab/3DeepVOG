import numpy as np
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
    N = N / np.linalg.norm(N, axis=0, keepdims=True)  # Normalize each vector
    phi = np.arctan2(N[2], N[0]) + np.pi / 2
    theta = np.arccos(N[1]) - np.pi/2
    return np.rad2deg(phi), np.rad2deg(theta)

def cart2sph_batch_LG(N):
    N = N / np.linalg.norm(N, axis=0, keepdims=True)  # Normalize each vector
    phi = -(np.arctan2(N[2], N[0]) + np.pi / 2)    
    theta = -(np.arctan2(N[2], N[1]) - np.pi / 2)
    # theta = -(np.arccos(N[1]) - np.pi/2)
    return np.rad2deg(phi), np.rad2deg(theta)
