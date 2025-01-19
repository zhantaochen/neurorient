import torch
import torch.nn as nn

import numpy as np
import e3nn
from e3nn import o3

import torchvision.models.resnet as resnet
import healpy as hp

from pytorch3d.transforms import matrix_to_euler_angles, quaternion_to_matrix

def so3_healpix_grid(rec_level: int=3):
    """Returns healpix grid over so3 of equally spaced rotations
   
    https://github.com/google-research/google-research/blob/4808a726f4b126ea38d49cdd152a6bb5d42efdf0/implicit_pdf/models.py#L272
    alpha: 0-2pi around Y
    beta: 0-pi around X
    gamma: 0-2pi around Y
    rec_level | num_points | bin width (deg)
    ----------------------------------------
         0    |         72 |    60
         1    |        576 |    30
         2    |       4608 |    15
         3    |      36864 |    7.5
         4    |     294912 |    3.75
         5    |    2359296 |    1.875
         
    :return: tensor of shape (3, npix)
    """
    n_side = 2**rec_level
    npix = hp.nside2npix(n_side)
    beta, alpha = hp.pix2ang(n_side, torch.arange(npix))
    gamma = torch.linspace(0, 2*np.pi, 6*n_side + 1)[:-1]

    alpha = alpha.repeat(len(gamma))
    beta = beta.repeat(len(gamma))
    gamma = torch.repeat_interleave(gamma, npix)
    return torch.stack((alpha, beta, gamma)).float()

def s2_irreps(lmax):
  return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])

def so3_irreps(lmax):
  return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])

def flat_wigner(lmax, alpha, beta, gamma):
  return torch.cat([
    (2 * l + 1) ** 0.5 * o3.wigner_D(l, alpha, beta, gamma).flatten(-2) for l in range(lmax + 1)
  ], dim=-1)

def so3_near_identity_grid(max_beta=np.pi / 8, max_gamma=2 * np.pi, n_alpha=8, n_beta=3, n_gamma=None):
  """Spatial grid over SO3 used to parametrize localized filter

  :return: rings of rotations around the identity, all points (rotations) in
           a ring are at the same distance from the identity
           size of the kernel = n_alpha * n_beta * n_gamma
  """
  if n_gamma is None:
      n_gamma = n_alpha 
  beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
  alpha = torch.linspace(0, 2 * np.pi, n_alpha)[:-1]
  pre_gamma = torch.linspace(-max_gamma, max_gamma, n_gamma)
  A, B, preC = torch.meshgrid(alpha, beta, pre_gamma, indexing="ij")
  C = preC - A
  A = A.flatten()
  B = B.flatten()
  C = C.flatten()
  return torch.stack((A, B, C))


class S2Conv(nn.Module):
  '''S2 group convolution which outputs signal over SO(3) irreps

  :f_in: feature dimensionality of input signal
  :f_out: feature dimensionality of output signal
  :lmax: maximum degree of harmonics used to represent input and output signals
         technically, you can have different degrees for input and output, but
         we do not explore that in our work
  :kernel_grid: spatial locations over which the filter is defined (alphas, betas)
                we find that it is better to parametrize filter in spatial domain
                and project to harmonics at every forward pass.
  '''
  def __init__(self, f_in: int, f_out: int, lmax: int, kernel_grid: tuple):
    super().__init__()
    # filter weight parametrized over spatial grid on S2
    self.register_parameter(
      "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
    )  # [f_in, f_out, n_s2_pts]

    # linear projection to convert filter weights to fourier domain
    self.register_buffer(
      "Y", o3.spherical_harmonics_alpha_beta(range(lmax + 1), *kernel_grid, normalization="component")
    )  # [n_s2_pts, (2*lmax+1)**2]

    # defines group convolution using appropriate irreps
    # note, we set internal_weights to False since we defined our own filter above
    self.lin = o3.Linear(s2_irreps(lmax), so3_irreps(lmax), 
                         f_in=f_in, f_out=f_out, internal_weights=False)

  def forward(self, x):
    '''Perform S2 group convolution to produce signal over irreps of SO(3).
    First project filter into fourier domain then perform convolution

    :x: tensor of shape (B, f_in, (2*lmax+1)**2), signal over S2 irreps
    :return: tensor of shape (B, f_out, sum_l^L (2*l+1)**2)
    '''
    psi = torch.einsum("ni,xyn->xyi", self.Y, self.w) / self.Y.shape[0] ** 0.5
    return self.lin(x, weight=psi)


class SO3Conv(nn.Module):
  '''SO3 group convolution

  :f_in: feature dimensionality of input signal
  :f_out: feature dimensionality of output signal
  :lmax: maximum degree of harmonics used to represent input and output signals
         technically, you can have different degrees for input and output, but
         we do not explore that in our work
  :kernel_grid: spatial locations over which the filter is defined (alphas, betas, gammas)
                we find that it is better to parametrize filter in spatial domain
                and project to harmonics at every forward pass
  '''
  def __init__(self, f_in: int, f_out: int, lmax: int, kernel_grid: tuple):
    super().__init__()

    # filter weight parametrized over spatial grid on SO3
    self.register_parameter(
      "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
    )  # [f_in, f_out, n_so3_pts]

    # wigner D matrices used to project spatial signal to irreps of SO(3)
    self.register_buffer("D", flat_wigner(lmax, *kernel_grid))  # [n_so3_pts, sum_l^L (2*l+1)**2]

    # defines group convolution using appropriate irreps
    self.lin = o3.Linear(so3_irreps(lmax), so3_irreps(lmax), 
                         f_in=f_in, f_out=f_out, internal_weights=False)

  def forward(self, x):
    '''Perform SO3 group convolution to produce signal over irreps of SO(3).
    First project filter into fourier domain then perform convolution

    :x: tensor of shape (B, f_in, sum_l^L (2*l+1)**2), signal over SO3 irreps
    :return: tensor of shape (B, f_out, sum_l^L (2*l+1)**2)
    '''
    psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
    return self.lin(x, weight=psi)

class ImageEncoder(nn.Module):
    def __init__(self,
                 size: int=18,
                 pretrained: bool=False,
                 pool_features: bool=False,
                 input_size: int=224,
                 **kwargs,
                 ):
        super().__init__()
        weights = 'DEFAULT' if pretrained else None
        full_model = eval(f'resnet.resnet{size}')(weights=weights)
        # Average the weights in the input channels...
        conv1_weight = full_model.conv1.weight.data.mean(dim = 1, keepdim = True)
        full_model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        full_model.conv1.weight.data = conv1_weight

        out_fdim = 512 if size in (18, 34) else 2048

        # remove pool and linear
        modules = list(full_model.children())[:-2]

        self.layers = nn.Sequential(*modules)

        if pool_features:
            modules.append(nn.AdaptiveAvgPool2d(1))
        #     self.output_shape = (out_fdim, 1, 1)
        # else:
        #     self.output_shape = (out_fdim, 7, 7)
        self.output_shape = self.layers(torch.zeros(1, 1, input_size, input_size)).shape[-3:]
    def forward(self, img):
        return self.layers(img)

def s2_healpix_grid(rec_level: int=0, max_beta: float=np.pi/6):
    """Returns healpix grid up to a max_beta
    """
    n_side = 2**rec_level
    npix = hp.nside2npix(n_side)
    m = hp.query_disc(nside=n_side, vec=(0,0,1), radius=max_beta)
    beta, alpha = hp.pix2ang(n_side, m)
    alpha = torch.from_numpy(alpha)
    beta = torch.from_numpy(beta)
    return torch.stack((alpha, beta)).float()


class Image2SphereProjector(nn.Module):
  '''Define orthographic projection from image space to half of sphere, returning
  coefficients of spherical harmonics

  :fmap_shape: shape of incoming feature map (channels, height, width)
  :fdim_sphere: dimensionality of featuremap projected to sphere
  :lmax: maximum degree of harmonics
  :coverage: fraction of feature map that is projected onto sphere
  :sigma: stdev of gaussians used to sample points in image space
  :max_beta: maximum azimuth angle projected onto sphere (np.pi/2 corresponds to half sphere)
  :taper_beta: if less than max_beta, taper magnitude of projected features beyond this angle
  :rec_level: recursion level of healpy grid where points are projected
  :n_subset: number of grid points used to perform projection, acts like dropout regularizer
  '''
  def __init__(self,
               fmap_shape, 
               sphere_fdim: int,
               lmax: int,
               coverage: float = 0.9,
               sigma: float = 0.2,
               max_beta: float = np.radians(90),
               taper_beta: float = np.radians(75),
               rec_level: int = 2,
               n_subset: int = 20,
              ):
    super().__init__()
    self.lmax = lmax
    self.n_subset = n_subset

    # point-wise linear operation to convert to proper dimensionality if needed
    if fmap_shape[0] != sphere_fdim:
      self.conv1x1 = nn.Conv2d(fmap_shape[0], sphere_fdim, 1)
    else:
      self.conv1x1 = nn.Identity()

    # determine sampling locations for orthographic projection
    self.kernel_grid = s2_healpix_grid(max_beta=max_beta, rec_level=rec_level)
    self.xyz = o3.angles_to_xyz(*self.kernel_grid)

    # orthographic projection
    max_radius = torch.linalg.norm(self.xyz[:,[0,2]], dim=1).max()
    sample_x = coverage * self.xyz[:,2] / max_radius # range -1 to 1
    sample_y = coverage * self.xyz[:,0] / max_radius

    gridx, gridy = torch.meshgrid(2*[torch.linspace(-1, 1, fmap_shape[1])], indexing='ij')
    scale = 1 / np.sqrt(2 * np.pi * sigma**2)
    data = scale * torch.exp(-((gridx.unsqueeze(-1) - sample_x).pow(2) \
                                +(gridy.unsqueeze(-1) - sample_y).pow(2)) / (2*sigma**2) )
    data = data / data.sum((0,1), keepdims=True)

    # apply mask to taper magnitude near border if desired
    betas = self.kernel_grid[1]
    if taper_beta < max_beta:
        mask = ((betas - max_beta)/(taper_beta - max_beta)).clamp(max=1).view(1, 1, -1)
    else:
        mask = torch.ones_like(data)

    data = (mask * data).unsqueeze(0).unsqueeze(0).to(torch.float32)
    self.weight = nn.Parameter(data= data, requires_grad=True)

    self.n_pts = self.weight.shape[-1]
    self.ind = torch.arange(self.n_pts)

    self.register_buffer(
        "Y", o3.spherical_harmonics_alpha_beta(range(lmax+1), *self.kernel_grid, normalization='component')
    )

  def forward(self, x):
    '''
    :x: float tensor of shape (B, C, H, W)
    :return: feature vector of shape (B,P,C) where P is number of points on S2
    '''
    x = self.conv1x1(x)

    if self.n_subset is not None:
        self.ind = torch.randperm(self.n_pts)[:self.n_subset]

    x = (x.unsqueeze(-1) * self.weight[..., self.ind]).sum((2,3))
    x = torch.relu(x)
    x = torch.einsum('ni,xyn->xyi', self.Y[self.ind], x) / self.ind.shape[0]**0.5
    return x

def compute_trace(rotA, rotB):
    '''
    rotA, rotB are tensors of shape (*,3,3)
    returns Tr(rotA, rotB.T)
    '''
    prod = torch.matmul(rotA, rotB.transpose(-1, -2))
    trace = prod.diagonal(dim1=-1, dim2=-2).sum(-1)
    return trace

def rotation_error(rotA, rotB):
    '''
    rotA, rotB are tensors of shape (*,3,3)
    returns rotation error in radians, tensor of shape (*)
    '''
    trace = compute_trace(rotA, rotB)
    return torch.arccos(torch.clamp( (trace - 1)/2, -1, 1))

def nearest_rotmat(src, target):
    '''return index of target that is nearest to each element in src
    uses negative trace of the dot product to avoid arccos operation
    :src: tensor of shape (B, 3, 3)
    :target: tensor of shape (*, 3, 3)
    '''
    trace = compute_trace(src.unsqueeze(1), target.unsqueeze(0))

    return torch.max(trace, dim=1)[1]
def s2_irreps(lmax):
  return o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])

def so3_irreps(lmax):
  return o3.Irreps([(2 * l + 1, (l, 1)) for l in range(lmax + 1)])

def flat_wigner(lmax, alpha, beta, gamma):
  return torch.cat([
    (2 * l + 1) ** 0.5 * o3.wigner_D(l, alpha, beta, gamma).flatten(-2) for l in range(lmax + 1)
  ], dim=-1)

def so3_near_identity_grid(max_beta=np.pi / 8, max_gamma=2 * np.pi, n_alpha=8, n_beta=3, n_gamma=None):
  """Spatial grid over SO3 used to parametrize localized filter

  :return: rings of rotations around the identity, all points (rotations) in
           a ring are at the same distance from the identity
           size of the kernel = n_alpha * n_beta * n_gamma
  """
  if n_gamma is None:
      n_gamma = n_alpha 
  beta = torch.arange(1, n_beta + 1) * max_beta / n_beta
  alpha = torch.linspace(0, 2 * np.pi, n_alpha)[:-1]
  pre_gamma = torch.linspace(-max_gamma, max_gamma, n_gamma)
  A, B, preC = torch.meshgrid(alpha, beta, pre_gamma, indexing="ij")
  C = preC - A
  A = A.flatten()
  B = B.flatten()
  C = C.flatten()
  return torch.stack((A, B, C))


class S2Conv(nn.Module):
  '''S2 group convolution which outputs signal over SO(3) irreps

  :f_in: feature dimensionality of input signal
  :f_out: feature dimensionality of output signal
  :lmax: maximum degree of harmonics used to represent input and output signals
         technically, you can have different degrees for input and output, but
         we do not explore that in our work
  :kernel_grid: spatial locations over which the filter is defined (alphas, betas)
                we find that it is better to parametrize filter in spatial domain
                and project to harmonics at every forward pass.
  '''
  def __init__(self, f_in: int, f_out: int, lmax: int, kernel_grid: tuple):
    super().__init__()
    # filter weight parametrized over spatial grid on S2
    self.register_parameter(
      "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
    )  # [f_in, f_out, n_s2_pts]

    # linear projection to convert filter weights to fourier domain
    self.register_buffer(
      "Y", o3.spherical_harmonics_alpha_beta(range(lmax + 1), *kernel_grid, normalization="component")
    )  # [n_s2_pts, (2*lmax+1)**2]

    # defines group convolution using appropriate irreps
    # note, we set internal_weights to False since we defined our own filter above
    self.lin = o3.Linear(s2_irreps(lmax), so3_irreps(lmax), 
                         f_in=f_in, f_out=f_out, internal_weights=False)

  def forward(self, x):
    '''Perform S2 group convolution to produce signal over irreps of SO(3).
    First project filter into fourier domain then perform convolution

    :x: tensor of shape (B, f_in, (2*lmax+1)**2), signal over S2 irreps
    :return: tensor of shape (B, f_out, sum_l^L (2*l+1)**2)
    '''
    psi = torch.einsum("ni,xyn->xyi", self.Y, self.w) / self.Y.shape[0] ** 0.5
    return self.lin(x, weight=psi)


class SO3Conv(nn.Module):
  '''SO3 group convolution

  :f_in: feature dimensionality of input signal
  :f_out: feature dimensionality of output signal
  :lmax: maximum degree of harmonics used to represent input and output signals
         technically, you can have different degrees for input and output, but
         we do not explore that in our work
  :kernel_grid: spatial locations over which the filter is defined (alphas, betas, gammas)
                we find that it is better to parametrize filter in spatial domain
                and project to harmonics at every forward pass
  '''
  def __init__(self, f_in: int, f_out: int, lmax: int, kernel_grid: tuple):
    super().__init__()

    # filter weight parametrized over spatial grid on SO3
    self.register_parameter(
      "w", torch.nn.Parameter(torch.randn(f_in, f_out, kernel_grid.shape[1]))
    )  # [f_in, f_out, n_so3_pts]

    # wigner D matrices used to project spatial signal to irreps of SO(3)
    self.register_buffer("D", flat_wigner(lmax, *kernel_grid))  # [n_so3_pts, sum_l^L (2*l+1)**2]

    # defines group convolution using appropriate irreps
    self.lin = o3.Linear(so3_irreps(lmax), so3_irreps(lmax), 
                         f_in=f_in, f_out=f_out, internal_weights=False)

  def forward(self, x):
    '''Perform SO3 group convolution to produce signal over irreps of SO(3).
    First project filter into fourier domain then perform convolution

    :x: tensor of shape (B, f_in, sum_l^L (2*l+1)**2), signal over SO3 irreps
    :return: tensor of shape (B, f_out, sum_l^L (2*l+1)**2)
    '''
    psi = torch.einsum("ni,xyn->xyi", self.D, self.w) / self.D.shape[0] ** 0.5
    return self.lin(x, weight=psi)
  
def get_rotation_grids(rec_level=2, lmax=6):
   
    output_xyx = so3_healpix_grid(rec_level=rec_level)
    output_wigners = flat_wigner(lmax, *output_xyx).transpose(0,1)
    output_rotmats = o3.angles_to_matrix(*output_xyx)
    return {
        "output_xyx": output_xyx,
        "output_wigners": output_wigners,
        "output_rotmats": output_rotmats,
    }

def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)

def random_rotations(n, seed=None):
    if seed is None:
      o = torch.randn((n, 4))
    else:
      rng = np.random.default_rng(42)
      o = torch.from_numpy(rng.normal(size=(n, 4)))
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]

    rotmats = quaternion_to_matrix(o)
    return rotmats

class I2S(nn.Module):
  '''
  Instantiate I2S-style network for predicting distributions over SO(3) from
  single image
  '''
  def __init__(self, lmax=6, s2_fdim=512, so3_fdim=16, rec_level=3, input_size=224):
    super().__init__()
    self.encoder = ImageEncoder(input_size=input_size)

    self.projector = Image2SphereProjector(
        fmap_shape=self.encoder.output_shape,
        sphere_fdim=s2_fdim,
        lmax=lmax,
    )

    # s2 filter has global support
    s2_kernel_grid = s2_healpix_grid(max_beta=np.inf, rec_level=1)
    self.s2_conv = S2Conv(s2_fdim, so3_fdim, lmax, s2_kernel_grid)

    self.so3_act = e3nn.nn.SO3Activation(lmax, lmax, act=torch.relu, resolution=10)

    # locally supported so3 filter
    so3_kernel_grid = so3_near_identity_grid()
    self.so3_conv = SO3Conv(so3_fdim, 1, lmax, so3_kernel_grid)

    # define spatial grid used to convert output irreps into valid prob distribution
    # we use rec_level=2 which corresponds to ~5000 points, which we find is
    # sufficient for training.  Using denser grids will slow down loss computation
    # output_xyx = so3_healpix_grid(rec_level=rec_level)
    # self.register_buffer(
    #     "output_wigners", flat_wigner(lmax, *output_xyx).transpose(0,1)
    # )
    # self.register_buffer(
    #     "output_rotmats", o3.angles_to_matrix(*output_xyx)
    # )
    self.set_rotation_grids(rec_level=rec_level, lmax=lmax)
  
  def set_rotation_grids(self, rec_level=2, lmax=6):
    output_xyx = so3_healpix_grid(rec_level=rec_level)
    random_xyx = matrix_to_euler_angles(random_rotations(output_xyx.size(1)//4, seed=42), 'XYX').T.to(output_xyx)
    output_xyx = torch.cat([output_xyx, random_xyx], dim=1)
    self.register_buffer(
        "output_wigners", flat_wigner(lmax, *output_xyx).transpose(0,1)
    )
    self.register_buffer(
        "output_rotmats", o3.angles_to_matrix(*output_xyx)
    )
    print(f"generated {output_xyx.size(1)} rotations")
     
  
  def forward(self, x):
    '''Returns so3 irreps

    :x: image, tensor of shape (B, 3, 224, 224)
    '''
    x = self.encoder(x)
    x = self.projector(x)
    x = self.s2_conv(x)
    x = self.so3_act(x)
    x = self.so3_conv(x)
    return x
  
  def compute_loss(self, img, gt_rot):
    '''Compute cross entropy loss using ground truth rotation, the correct label
    is the nearest rotation in the spatial grid to the ground truth rotation

    :img: float tensor of shape (B, 3, 224, 224)
    :gt_rotation: valid rotation matrices, tensor of shape (B, 3, 3)
    '''
    x = self.forward(img)
    grid_signal = torch.matmul(x, self.output_wigners).squeeze(1)
    rotmats = self.output_rotmats

    # find nearest grid point to ground truth rotation matrix
    rot_id = nearest_rotmat(gt_rot, rotmats)
    loss = nn.CrossEntropyLoss()(grid_signal, rot_id)

    with torch.no_grad():
        pred_id = grid_signal.max(dim=1)[1]
        pred_rotmat = rotmats[pred_id]
        acc = rotation_error(gt_rot, pred_rotmat)

    return loss, acc.cpu().numpy()


  def compute_logits(self, img, wigners=None, require_grad=True):
    '''Computes probability distribution over arbitrary spatial grid specified by
    wigners

    Our method can be trained on a sparser spatial resolution, but queried at a much denser
    resolution (up to rec_level=5)
    '''
    if wigners is None:
        wigners = self.output_wigners

    if require_grad:
       probs = self._compute_logits(img, wigners)
    else:
        with torch.no_grad():
            probs = self._compute_logits(img, wigners)
    return probs

  def _compute_logits(self, img, wigners):
    '''Computes probability distribution over arbitrary spatial grid specified by
    wigners

    Our method can be trained on a sparser spatial resolution, but queried at a much denser
    resolution (up to rec_level=5)
    '''
    x = self.forward(img)
    logits = torch.matmul(x, wigners).squeeze(1)
    return logits
  
  def compute_probabilities(self, *args, **kwargs):
    return self.compute_logits(*args, **kwargs).softmax(dim=-1)
