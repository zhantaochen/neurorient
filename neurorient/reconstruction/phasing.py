"""
adapted from https://gitlab.osti.gov/mtip/spinifel/-/blob/development/spinifel/sequential/phasing.py
"""

import torch
try:
    import cupy as xp
    from cupyx.scipy.ndimage import gaussian_filter
    using_cupy = True
except ImportError:
    import numpy as xp
    from scipy.ndimage import gaussian_filter
    using_cupy = False
from tqdm import tqdm

def center_of_mass(rho_, hkl_, M):
    """
    Compute the object's center of mass.

    :param rho_: electron density (fftshifted)
    :param hkl_: coordinates
    :param M: cubic length of electron density volume
    :return vect: vector center of mass in units of pixels
    """
    rho_ = torch.abs(rho_)
    num = torch.sum(rho_ * hkl_, dim=(1, 2, 3))
    den = rho_.sum() + torch.finfo(rho_.dtype).eps
    return torch.round(num / den * M / 2)

def recenter(rho_, support_, M):
    """
    Shift center of the electron density and support to origin.

    :param rho_: electron density (fftshifted)
    :param support_: object's support
    :param M: cubic length of electron density volume
    """
    ls = torch.linspace(-1, 1, M + 1)
    ls = (ls[:-1] + ls[1:]) / 2

    hkl_list = torch.meshgrid(ls, ls, ls)
    hkl_ = torch.stack([torch.fft.fftshift(coord).to(rho_) for coord in hkl_list])

    k = 0
    vect = torch.ones(3).to(rho_.device) * 3
    while torch.any(vect.abs() > 2):
        print(k, vect, torch.any(vect.abs() > 2))
        vect = center_of_mass(rho_, hkl_, M)
        print(k, vect, torch.any(vect > 2))
        for i in range(3):
            shift = int(vect[i].item())
            rho_[:] = torch.roll(rho_, -shift, dims=i)
            support_[:] = torch.roll(support_, -shift, dims=i)
        k += 1
        if k > 10:
            break
    return rho_, support_

def shrink_wrap(sigma, rho_, support_, method=None, weight=1.0, cutoff=0.05):
    """
    Perform shrinkwrap operation to update the support for convergence.

    :param sigma: Gaussian standard deviation to low-pass filter density with
    :param rho_: electron density estimate
    :param support_: object support
    :param method: {'max', 'std'}, default: std
    kwargs:
    :param cutoff: method='max', threshold as a fraction of maximum density value
    :param weight: method='std', threshold as standard deviation of density times a weight factor
    """
    rho_abs_ = rho_.abs().detach().cpu().numpy()
    # By using 'wrap', we don't need to fftshift it back and forth
    if using_cupy:
        rho_gauss_ = gaussian_filter(xp.array(rho_abs_), mode="wrap", sigma=sigma, truncate=2).get()
    else:
        rho_gauss_ = gaussian_filter(rho_abs_, mode="wrap", sigma=sigma, truncate=2)
    rho_gauss_ = torch.from_numpy(rho_gauss_).to(rho_.device)
    
    if method == None:
        method = "std"
    if method == "std":
        threshold = torch.std(rho_gauss_) * weight
        support_[:] = rho_gauss_ > threshold
    elif method == "max":
        threshold = rho_abs_.max() * cutoff * weight
        support_[:] = rho_gauss_ > threshold
    elif method == "min_max":
        threshold_low = rho_abs_.max() * cutoff
        # support_[:] = rho_gauss_ > threshold_low
        # threshold_high = torch.mean(rho_gauss_) + 3 * torch.std(rho_gauss_)
        threshold_high = rho_abs_.max() * (1-cutoff)
        support_[:] = torch.logical_and(
            rho_gauss_ > threshold_low, rho_gauss_ < threshold_high
        )
    else:
        raise ValueError(f"Invalid method: {method}. Options are 'std' or 'max'.")
    
    return support_

def create_support_(amplitude_, M, support_threshold=0.01):
    sl = slice(M // 4, -(M // 4))
    square_support = torch.zeros((M,)*3).to(amplitude_.device)
    square_support[sl, sl, sl] = 1
    square_support_ = torch.fft.ifftshift(square_support)
    thresh_support_ = amplitude_ > support_threshold * amplitude_.max()
    return torch.logical_and(square_support_, thresh_support_)

def step_ER(rho_, amplitude_, amp_mask_, support_, rho_max):
    """
    Perform Error Reduction (ER) operation by updating the amplitudes from the current electron density estimtate
    with those computed from the autocorrelation, and enforcing electron density to be positive (real space constraint).

    :param rho_: current electron density estimate
    :param amplitude_: amplitudes computed from the autocorrelation
    :param amp_mask_: amplitude mask
    :param support_: binary mask for object's support
    :param rho_mask: maximum permitted electron density value
    """
    rho_mod_, support_star_ = step_phase(rho_, amplitude_, amp_mask_, support_)
    rho_[:] = torch.where(support_star_, rho_mod_, 0)
    i_overmax = rho_mod_ > rho_max
    rho_[i_overmax] = rho_max
    return rho_

def step_DM(beta, rho_, amplitude_, amp_mask_, support_, rho_max):
    """
    Perform Difference Map (DM) operation. The DM algorithm refines the solution by alternating between real and 
    Fourier space constraints and nudging the solution towards an intersection of both constraints.

    :param beta: DM constant
    :param rho_: electron density estimate
    :param amplitude_: amplitudes computed from the autocorrelation
    :param amp_mask_: amplitude mask
    :param support_: binary mask for object's support
    :param rho_max: maximum permitted electron density value
    """
    # First, apply Fourier space constraint (P1) then real space constraint (P2)
    rho_mod_1, _ = step_phase(rho_, amplitude_, amp_mask_, support_)

    # Now, apply real space constraint (P2) then Fourier space constraint (P1)
    rho_mod_2, _ = step_phase(torch.where(support_, rho_mod_1, rho_), amplitude_, amp_mask_, support_)

    # DM update
    rho_dm = rho_ + beta * (rho_mod_1 - rho_mod_2)

    # Ensure the real-space constraints are satisfied
    rho_[:] = torch.where(support_, rho_dm, 0)
    i_overmax = rho_dm > rho_max
    rho_[i_overmax] = rho_max
    
    return rho_



def step_HIO(beta, rho_, amplitude_, amp_mask_, support_, rho_max):
    """
    Perform Hybrid-Input-Output (HIO) operation by updating the amplitudes from the current electron density estimtate
    with those computed from the autocorrelation, and using negative feedback in Fourier space in order to progressively
    force the solution to conform to the Fourier domain constraints (support).

    :param beta: feedback constant
    :param rho_: electron density estimate
    :param amplitude_: amplitudes computed from the autocorrelation
    :param amp_mask_: amplitude mask
    :param support_: binary mask for object's support
    :param rho_mask: maximum permitted electron density value
    """
    rho_mod_, support_star_ = step_phase(rho_, amplitude_, amp_mask_, support_)
    rho_[:] = torch.where(support_star_, rho_mod_, rho_ - beta * rho_mod_)
    i_overmax = rho_mod_ > rho_max
    rho_[i_overmax] += 2 * beta * rho_mod_[i_overmax] - rho_max
    return rho_

def step_phase(rho_, amplitude_, amp_mask_, support_):
    rho_hat_ = torch.fft.fftn(rho_)
    phases_ = torch.angle(rho_hat_)
    rho_hat_mod_ = torch.where(amp_mask_, amplitude_ * torch.exp(1j * phases_), rho_hat_)
    rho_mod_ = torch.fft.ifftn(rho_hat_mod_).real

    support_star_ = torch.logical_and(support_, rho_mod_>0)
    return rho_mod_, support_star_

class PhaseRetriever:
    
    def __init__(self, n_phase_loops: int=10, nER: int=50, nHIO: int=25, nDM: int=25, beta_HIO: float=0.9, beta_DM: float=1.0, support=None, shrink_wrap_method: str=None) -> None:
        self.n_phase_loops = n_phase_loops
        self.nER = nER
        self.nHIO = nHIO
        self.nDM = nDM
        self.beta_HIO = beta_HIO
        self.beta_DM = beta_DM
        self.support = support 
        self.shrink_wrap_method = shrink_wrap_method
    
    def ER_loop(self, n_loops, rho_, amplitude_, amp_mask_, support_, rho_max):
        for i in range(n_loops):
            rho_ = step_ER(rho_, amplitude_, amp_mask_, support_, rho_max)
        return rho_
        
    def HIO_loop(self, n_loops, beta, rho_, amplitude_, amp_mask_, support_, rho_max):
        for k in range(n_loops):
            rho_ = step_HIO(beta, rho_, amplitude_, amp_mask_, support_, rho_max)
        return rho_
    
    def DM_loop(self, n_loops, beta, rho_, amplitude_, amp_mask_, support_, rho_max):
        for k in range(n_loops):
            rho_ = step_DM(beta, rho_, amplitude_, amp_mask_, support_, rho_max)
        return rho_
    
    def phase(self, amplitude, rho=None, rho_max=torch.inf):
        
        device = amplitude.device
        amplitude_ = torch.fft.ifftshift(amplitude).detach()
        
        M = amplitude.shape[-1]
        amp_mask_ = torch.ones((M,)*3).to(torch.bool).to(device)
        if self.support is None:
            support_ = create_support_(amplitude_, M).to(device)
        else:
            support_ = torch.fft.ifftshift(self.support).to(device)
            
        if rho is None:
            rho_ = torch.rand((M,)*3).to(device)
        else:
            rho_ = torch.fft.ifftshift(rho).detach().to(device)
        
        with torch.no_grad():
            for i in tqdm(range(self.n_phase_loops), desc="Phase Retrieval"):
                rho_ = self.ER_loop(self.nER, rho_, amplitude_, amp_mask_, support_, rho_max)
                rho_ = rho_.clip_(0.)
                rho_ = self.HIO_loop(self.nHIO, self.beta_HIO, rho_, amplitude_, amp_mask_, support_, rho_max)
                rho_ = rho_.clip_(0.)
                
                rho_ = self.DM_loop(self.nDM, self.beta_DM, rho_, amplitude_, amp_mask_, support_, rho_max)
                rho_ = rho_.clip_(0.)
                
                rho_ = self.ER_loop(self.nER, rho_, amplitude_, amp_mask_, support_, rho_max)
                rho_ = rho_.clip_(0.)
                support_ = shrink_wrap(1, rho_, support_, method=self.shrink_wrap_method, weight=1.0, cutoff=0.05)
                rho_, support_ = recenter(rho_, support_, M)
            
            
            rho_ = self.ER_loop(self.nER, rho_, amplitude_, amp_mask_, support_, rho_max)
            rho_ = rho_.clip_(0.)
            
            # rho_ = self.HIO_loop(self.nHIO, self.beta_HIO, rho_, amplitude_, amp_mask_, support_, rho_max)
            # rho_ = rho_.clip_(0.)

            # rho_ = self.DM_loop(self.nDM, self.beta_DM, rho_, amplitude_, amp_mask_, support_, rho_max)
            # rho_ = rho_.clip_(0.)
            
        rho_, support_ = recenter(torch.nan_to_num(rho_), support_, M)
        rho_phased = torch.fft.fftshift(rho_)
        support_phased = torch.fft.fftshift(support_)
        
        # intensities_phased_ = torch.fft.fftn(rho_).abs().pow(2)
        # intensities_phased = torch.fft.fftshift(intensities_phased_)
        # ac_phased_ = torch.abs(torch.fft.ifftn(intensities_phased_))
        # ac_phased = torch.fft.fftshift(ac_phased_)
        
        return rho_phased, support_phased