# %%
# Importing necessary libraries
import os
import numpy as np
import torch
from tifffile import imread
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.simulations import ForwardModel
from VolumeRaytraceLFM.birefringence_implementations import BirefringentVolume
from VolumeRaytraceLFM.volumes import volume_args
from VolumeRaytraceLFM.setup_parameters import (
    setup_optical_parameters,
    setup_iteration_parameters,
)
from VolumeRaytraceLFM.reconstructions import ReconstructionConfig, Reconstructor
from VolumeRaytraceLFM.utils.file_utils import (
    create_unique_directory,
)
from VolumeRaytraceLFM.utils.file_utils import save_as_tif

# %%
# Configuration parameters
simulate = False

# Volume creation arguments, see src/VolumeRaytraceLFM/volumes/volume_args.py for more options
volume_gt_creation_args = volume_args.voxel_args
volume_initial_creation_args = volume_args.random_args1

# Paths to the optical and iteration configuration files
optical_config_file = os.path.join("..", "config", "Shell", "optical_config.json")
iter_config_file = os.path.join("..", "config", "Shell", "iter_config.json")

# Path to the directory where the reconstruction will be saved
recon_output_dir = os.path.join("..", "reconstructions", "Shell")

# Whether to continue a previous reconstruction or start from a given volume
continue_recon = True
recon_init_file_path = os.path.join(r"../data/Shell/Simulation Data/ShellSimulPaper1Noise5.h5")
# recon_init_file_path = os.path.join(r"../reconstructions/Shell/2026-04-30_12-33-09_Mac_ShellSimPap1_ShellInitRand1_OptInfDef_ItDef90/volume.h5")
# For loading forward images that were saved in a previous reconstruction folder

# measurement_dir = os.path.join(r"../data/2025_09/ClamShell/Experimental Data")
measurement_dir = os.path.join(r"../data/Shell/Simulation Data/LF Images BirTomo")

BACKEND = BackEnds.PYTORCH
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# %%
# Simulate or load forward images
optical_info = setup_optical_parameters(optical_config_file)

if simulate:
    optical_system = {"optical_info": optical_info}
    simulator = ForwardModel(optical_system, backend=BACKEND)
    volume_GT = BirefringentVolume(
        backend=BACKEND,
        optical_info=optical_info,
        volume_creation_args=volume_gt_creation_args,
    )
    simulator.forward_model(volume_GT)
    simulator.view_images()
    ret_image_meas = simulator.ret_img.detach().numpy()
    azim_image_meas = simulator.azim_img.detach().numpy()
else:
    ret_image_meas_tiff = imread(os.path.join(measurement_dir, "ShellSimulPaper1OptInfDef_Ret.tif"))
    ret_image_meas = np.array(ret_image_meas_tiff)
    azim_image_meas_tiff = imread(os.path.join(measurement_dir, "ShellSimulPaper1OptInfDef_Azim.tif"))
    azim_image_meas = np.array(azim_image_meas_tiff)
    volume_GT = None

# %%
# Run reconstruction
recon_optical_info = optical_info.copy()
iteration_params = setup_iteration_parameters(iter_config_file)
recon_dir_postfix = iteration_params["general"]["output_directory_postfix"]
recon_directory = create_unique_directory(recon_output_dir, postfix=recon_dir_postfix)
if continue_recon:
    initial_volume = BirefringentVolume.init_from_file(recon_init_file_path, BACKEND, recon_optical_info)
else:
    initial_volume = BirefringentVolume(
        backend=BACKEND,
        optical_info=recon_optical_info,
        volume_creation_args=volume_initial_creation_args,
    )
recon_config = ReconstructionConfig(
    recon_optical_info,
    ret_image_meas,
    azim_image_meas,
    initial_volume,
    iteration_params,
    gt_vol=volume_GT
)
recon_config.save(recon_directory)
reconstructor = Reconstructor(
    recon_config,
    output_dir=recon_directory,
    device=DEVICE
)
reconstructor.reconstruct()
print("Reconstruction complete")

 # %% 
# Save images as TIF
save_dir = r"/Users/rudolfo/Software/GitHub/BirTomo/data/2025_04/SpiculeA Experim&Simulation/Simulation Data/LF Images BirTomo"
save_as_tif(os.path.join(save_dir, "ret_image_meas.tif"),
            ret_image_meas,
            {"Optical info": optical_info}
            )
save_as_tif(os.path.join(save_dir, "azim_image_meas.tif"),
            azim_image_meas,
            {"Optical info": optical_info}
            )

# %%
