# `neurorient`: Self-supervised machine learning framework for x-ray single particle imaging reconstruction.

Code repository for “Augmenting x-ray single particle imaging reconstruction with self-supervised machine learning” (https://arxiv.org/abs/2311.16652).

## Key scripts
- `generate_diffraction_images.ipynb`: generating basic diffraction patterns based on PDB-format structure files.
- `train.py`: perform training of the model provided some input `*.yaml` configuration file.
    - Example usage: `python train.py --yaml_file yaml_template.yaml`.
- `compute_metrics.ipynb`: evaluate various aspects of model performance.
- `prepare_spinifel_inputs.ipynb`: prepare input files (`*.h5`) for `SpiniFEL` (https://gitlab.osti.gov/mtip/spinifel); in particular, please use the branch `network_prior` with commit `f4b95694ca92a378ef16bd88f91fe6be42ee050d` to best produce results presented in the paper.

## Key Dependencies

Key dependencies are listed below.

```
lightning==2.0.7
mrcfile==1.4.3
numba==0.57.1
numpy==1.23.5
pytorch3d==0.7.4
scipy==1.11.2
skopi==0.6.0
torch==2.0.1
torch-scatter==2.1.1
torchkbnufft==1.4.0
```
