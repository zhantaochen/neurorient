# Self-supervised machine learning for x-ray single particle imaging reconstruction.

This is the code repository for “Augmenting x-ray single particle imaging reconstruction with self-supervised machine learning” (https://arxiv.org/abs/2311.16652), which has been accepted by Newton.

## General workflow

### Data
- Use the `generate_diffraction_images.ipynb` to generate noise-free images for the chosen PDB (put the `*.pdb` file in `input/pdb`). Suppose the generated data is saved at `some_path/data`.
- Make a copy of the template `1bxr_resnet18_100x_coslr_fp.yaml` (change the filename if needed). Provide the following paths: `DATASET_DIRECTORY` (where to find training data), `BASE_DIRECTORY`, and `CHKPT_DIRECTORY` (where to save training logs and checkpoints, relative to `BASE_DIRECTORY`).
- The real experimental dataset can be prepared using the notebook `prepare_iUCrJ_PR772_dataset_downsampled.ipynb`; you can download the required data file `amo86615_PR772_7k_PCA_7303.h5` here: https://www-dev.cxidb.org/data/88/diffractionData/7k/.
    - More details about the used dataset can be found in the work by Rose et al. (IUCrJ, 2018; https://doi.org/10.1107/S205225251801120X).

### Training
- Run the `train.py` or `train_normal_enc_group_inv_dec.py` (provide config by `--yaml`) for training.
- An interactive notebook `train.ipynb` is also provided.


### Evaluation & Application
- Visualize the training results and make predictions with `compute_metrics.ipynb`. Expected results can be found in cell outputs of this notebook.
- In order to prepare input files for `SpiniFEL` (M-TIP algorithm implementation), you need to first run `prepare_spinifel_inputs.ipynb`. Please refer to the documentation of `SpiniFEL` to learn more details.
- To prepare input files for `Dragonfly` (EMC algorithm implementation):
    - First, run `dragonfly_comparison_generate_diffraction_images.ipynb` to generate diffraction patterns.
    - Then, run `prepare_dragonfly_inputs.ipynb` to prepare input file that can be parsed by `Dragonfly`.
    - Finally, run `dragonfly_comparison_generate_detector_file.ipynb` to prepare a detector file for `Dragonfly` based on `Skopi`'s detector. An example detector file (such as `det_sim_Newton_R1_300x300.h5`) needs to be prepared using `Dragonfly` before running this notebook.

## Key scripts
- `generate_diffraction_images.ipynb`: generating basic diffraction patterns based on PDB-format structure files.
- `train.py`: perform training of the model provided some input `*.yaml` configuration file.
    - Example usage: `python train.py --yaml_file yaml_template.yaml`.
- `train.ipynb`: essentially same as the `train.py` for the purpose of easier debugging and demonstration.
- `train_normal_enc_group_inv_dec.py`: perform training using a normal (same as the one used in `train.py`) neural network encoder but a group-invariant decoder for highly symmetric particles such as the PR772.
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