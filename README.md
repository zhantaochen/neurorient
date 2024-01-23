# `neurorient`: Self-supervised machine learning framework for x-ray single particle imaging reconstruction.

Code repository for “Augmenting x-ray single particle imaging reconstruction with self-supervised machine learning” (https://arxiv.org/abs/2311.16652).

## General workflow

- Use the `generate_diffraction_images.ipynb` to generate noise-free images for the chosen PDB (put the `*.pdb` file in `input/pdb`). Suppose the generated data is saved at `some_path/data`.
- Make a copy of the template `1bxr_resnet18_100x_coslr_fp.yaml` (change the filename if needed). Provide the following paths: `DATASET_DIRECTORY` (where to find training data), `BASE_DIRECTORY`, and `CHKPT_DIRECTORY` (where to save training logs and checkpoints, relative to `BASE_DIRECTORY`).
- Run the `train.py` (provide config by `--yaml`) or `train.ipynb` (provide config inside the notebook).
- Visualize the training results and make predictions with `compute_metrics.ipynb`. Expected results can be found in cell outputs of this notebook.
- In order to prepare input files for `SpiniFEL` (M-TIP algorithm package), you need to first run `prepare_spinifel_inputs.ipynb`. Please refer to the documentation of `SpiniFEL` to learn more details.

## Key scripts
- `generate_diffraction_images.ipynb`: generating basic diffraction patterns based on PDB-format structure files.
- `train.py`: perform training of the model provided some input `*.yaml` configuration file.
    - Example usage: `python train.py --yaml_file yaml_template.yaml`.
- `train.ipynb`: essentially same as the `train.py` for the purpose of easier debugging and demonstration.
- `compute_metrics.ipynb`: evaluate various aspects of model performance.
- `prepare_spinifel_inputs.ipynb`: prepare input files (`*.h5`) for `SpiniFEL` (https://gitlab.osti.gov/mtip/spinifel); in particular, please use the branch `network_prior` with commit `f4b95694ca92a378ef16bd88f91fe6be42ee050d` to best produce results presented in the paper.

## Installation and Key Dependencies

There is no installation for `neurorient` itself, you can directly download the source codes and run all the notebooks and python scripts.

The code has been tested on `SUSE Linux Enterprise Server 15 SP4`, which should generally work under most operating systems given that the following key dependencies are successfully installed:

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
