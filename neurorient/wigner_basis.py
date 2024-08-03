import numpy as np
import torch

from spherical import Wigner

class WignerD6Basis:
    def __init__(self,):
        self.wigner = Wigner(6,6)
        
    def m2i(self, m):
        return m + 6
    
    def i2m(self, i):
        return i - 6
        
    def evaluate_spha_coefficients(self, q):
        device = q.device
        dtype = q.dtype
        q = q.detach().cpu().numpy()
        D_mat = self.wigner.D(q).reshape(-1, 13, 13)
        D_out_complex = 1/5 * (
            - np.sqrt(7)  * D_mat[:,self.m2i( 5),:]
            + np.sqrt(11) * D_mat[:,self.m2i( 0),:]
            + np.sqrt(7)  * D_mat[:,self.m2i(-5),:]
        )
        D_out_complex = torch.from_numpy(D_out_complex).to(device)
        D_out = torch.zeros(q.shape[0], 13, device=device, dtype=dtype)
        D_out[:,:6] = np.sqrt(2) * D_out_complex.imag[:,:6]
        D_out[:,6]  = D_out_complex.real[:,6]
        D_out[:,7:] = np.sqrt(2) * D_out_complex.real[:,7:]
        return D_out