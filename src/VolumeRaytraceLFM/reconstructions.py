"""This module contains the ReconstructionConfig and Reconstructor classes."""

import sys
import copy
import time
import os
import json
import torch
import numpy as np
from tqdm import tqdm
import csv
import pickle
import tifffile
import matplotlib.pyplot as plt

## For analyzing the memory usage of a function
# from memory_profiler import profile
import gc
from VolumeRaytraceLFM.abstract_classes import BackEnds
from VolumeRaytraceLFM.birefringence_implementations import (
    BirefringentVolume,
    BirefringentRaytraceLFM,
)
from VolumeRaytraceLFM.visualization.plotting_ret_azim import (
    plot_retardance_orientation,
)
from VolumeRaytraceLFM.visualization.plotting_volume import (
    convert_volume_to_2d_mip,
    prepare_plot_mip,
)
from VolumeRaytraceLFM.visualization.plt_util import setup_visualization
from VolumeRaytraceLFM.visualization.plotting_iterations import (
    plot_iteration_update_gridspec,
)
from VolumeRaytraceLFM.utils.file_utils import create_unique_directory
from VolumeRaytraceLFM.utils.dimensions_utils import (
    get_region_of_ones_shape,
    reshape_and_crop,
    store_as_pytorch_parameter,
)
from VolumeRaytraceLFM.utils.error_handling import (
    check_for_inf_or_nan,
    check_for_negative_values,
    check_for_negative_values_dict,
)
from VolumeRaytraceLFM.utils.json_utils import ComplexArrayEncoder
from VolumeRaytraceLFM.metrics.metric import PolarimetricLossFunction
from VolumeRaytraceLFM.utils.optimizer_utils import calculate_adjusted_lr, print_moments
from VolumeRaytraceLFM.volumes.optic_axis import (
    fill_vector_based_on_nonaxial,
    stay_on_sphere,
    spherical_to_unit_vector_torch,
)
from VolumeRaytraceLFM.utils.mask_utils import filter_voxels_using_retardance
from VolumeRaytraceLFM.nerf import setup_optimizer_nerf, predict_voxel_properties


DEBUG = False
PRINT_GRADIENTS = False
PRINT_TIMING_INFO = False
CLIP_GRADIENT_NORM = False

if DEBUG:
    print("Debug mode is on.")
    from VolumeRaytraceLFM.utils.dict_utils import (
        extract_numbers_from_dict_of_lists,
        transform_dict_list_to_set,
    )


class ReconstructionConfig:
    def __init__(
        self,
        optical_info,
        ret_image,
        azim_image,
        initial_vol,
        iteration_params,
        loss_fcn=None,
        gt_vol=None,
        intensity_img_list=None,
    ):
        """
        Initialize the ReconstructorConfig with the provided parameters.

        optical_info: The optical parameters for the reconstruction process.
        retardance_image: Measured retardance image.
        azimuth_image: Measured azimuth image.
        initial_volume: An initial estimation of the volume.
        """
        start_time = time.perf_counter()
        assert isinstance(
            optical_info, dict
        ), "Expected optical_info to be a dictionary"
        assert isinstance(
            ret_image, (torch.Tensor, np.ndarray)
        ), "Expected ret_image to be a PyTorch Tensor or a numpy array"
        assert isinstance(
            azim_image, (torch.Tensor, np.ndarray)
        ), "Expected azim_image to be a PyTorch Tensor or a numpy array"
        assert isinstance(
            initial_vol, BirefringentVolume
        ), "Expected initial_volume to be of type BirefringentVolume"
        assert isinstance(
            iteration_params, dict
        ), "Expected iteration_params to be a dictionary"
        if intensity_img_list:
            assert isinstance(
                intensity_img_list, list
            ), "Expected intensity_img_list to be a list"
            for img in intensity_img_list:
                assert isinstance(
                    img, (torch.Tensor, np.ndarray)
                ), "Each image in intensity_img_list should be a Tensor or ndarray"
        if loss_fcn:
            assert callable(loss_fcn), "Expected loss_function to be callable"
        if gt_vol:
            assert isinstance(
                gt_vol, BirefringentVolume
            ), "Expected gt_vol to be of type BirefringentVolume"

        self.optical_info = optical_info
        self.retardance_image = self._to_numpy(ret_image)
        self.azimuth_image = self._to_numpy(azim_image)
        radiometry_path = iteration_params.get("radiometry_path", None)
        if radiometry_path:
            self.radiometry = tifffile.imread(radiometry_path)
        else:
            self.radiometry = None
        self.initial_volume = initial_vol
        self.interation_parameters = iteration_params
        self.loss_function = loss_fcn
        self.gt_volume = gt_vol
        self.intensity_img_list = (
            [self._to_numpy(img) for img in intensity_img_list]
            if intensity_img_list
            else None
        )
        self.ret_img_pred = None
        self.azim_img_pred = None
        self.recon_directory = None
        end_time = time.perf_counter()
        print(
            f"ReconstructionConfig initialized in {end_time - start_time:.2f} seconds"
        )

    def _to_numpy(self, image):
        """Convert image to a numpy array, if it's not already."""
        if isinstance(image, torch.Tensor):
            return image.detach().cpu().numpy()
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise TypeError("Image must be a PyTorch Tensor or a numpy array")

    def save(self, parent_directory):
        """Save the ReconstructionConfig to the specified directory.
        Args:
            parent_directory (str): Path to the directory where the
                config_parameters directory will be created.
        Returns:
            None
        Class instance attibutes saved:
            - self.optical_info
            - self.retardance_image
            - self.azimuth_image
            - self.interation_parameters
        (if available)
            - self.initial_volume
            - self.gt_volume
        Class instance attributes modified:
            - self.recon_directory
        """
        start_time = time.perf_counter()
        self.recon_directory = parent_directory

        directory = os.path.join(parent_directory, "config_parameters")
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Save the retardance and azimuth images
        np.save(os.path.join(directory, "ret_image.npy"), self.retardance_image)
        np.save(os.path.join(directory, "azim_image.npy"), self.azimuth_image)
        if self.radiometry is not None:
            np.save(os.path.join(directory, "radiometry"), self.radiometry)
        plt.ioff()
        my_fig = plot_retardance_orientation(
            self.retardance_image, self.azimuth_image, "hsv", include_labels=True
        )
        my_fig.savefig(
            os.path.join(directory, "ret_azim.png"), bbox_inches="tight", dpi=300
        )
        plt.close(my_fig)
        with open(os.path.join(directory, "optical_info.json"), "w") as f:
            json.dump(self.optical_info, f, indent=4, cls=ComplexArrayEncoder)
        with open(os.path.join(directory, "iteration_params.json"), "w") as f:
            json.dump(self.interation_parameters, f, indent=4)
        # Save the volumes if the 'save_as_file' method exists
        if hasattr(self.initial_volume, "save_as_file"):
            my_description = "Initial volume used for reconstruction."
            self.initial_volume.save_as_file(
                os.path.join(directory, "initial_volume.h5"), description=my_description
            )
        if self.gt_volume and hasattr(self.gt_volume, "save_as_file"):
            my_description = "Ground truth volume used for reconstruction."
            self.gt_volume.save_as_file(
                os.path.join(directory, "gt_volume.h5"), description=my_description
            )
        end_time = time.perf_counter()
        print(f"ReconstructionConfig saved in {end_time - start_time:.2f} seconds")

    @classmethod
    def load(cls, parent_directory):
        """Load the ReconstructionConfig from the specified directory."""
        directory = os.path.join(parent_directory, "config_parameters")
        # Load the numpy arrays
        ret_image = np.load(os.path.join(directory, "ret_image.npy"))
        azim_image = np.load(os.path.join(directory, "azim_image.npy"))
        # Load the dictionaries
        with open(os.path.join(directory, "optical_info.json"), "r") as f:
            optical_info = json.load(f)
        with open(os.path.join(directory, "iteration_params.json"), "r") as f:
            iteration_params = json.load(f)
        # Initialize the initial_volume and gt_volume from files or set to None if files don't exist
        initial_volume_file = os.path.join(directory, "initial_volume.h5")
        gt_volume_file = os.path.join(directory, "gt_volume.h5")
        if os.path.exists(initial_volume_file):
            initial_volume = BirefringentVolume.load_from_file(
                initial_volume_file, backend_type="torch"
            )
        else:
            initial_volume = None
        if os.path.exists(gt_volume_file):
            gt_volume = BirefringentVolume.load_from_file(
                gt_volume_file, backend_type="torch"
            )
        else:
            gt_volume = None
        # The loss_function is not saved and should be redefined
        loss_fcn = None
        return cls(
            optical_info,
            ret_image,
            azim_image,
            initial_volume,
            iteration_params,
            loss_fcn=loss_fcn,
            gt_vol=gt_volume,
        )


class Reconstructor:
    backend = BackEnds.PYTORCH

    def __init__(
        self,
        recon_info: ReconstructionConfig,
        output_dir=None,
        device="cpu",
        omit_rays_based_on_pixels=False,
        apply_volume_mask=False,
    ):
        """
        Initialize the Reconstructor with the provided parameters.

        recon_info (class): containing reconstruction parameters
        """
        start_time = time.perf_counter()
        print(f"\nInitializing a Reconstructor, using computing device {device}")
        self.optical_info = recon_info.optical_info
        self.ret_img_meas = recon_info.retardance_image
        self.azim_img_meas = recon_info.azimuth_image
        # if initial_volume is not None else self._initialize_volume()
        self.volume_initial_guess = recon_info.initial_volume
        self.iteration_params = recon_info.interation_parameters
        self.volume_ground_truth = recon_info.gt_volume
        self.intensity_imgs_meas = recon_info.intensity_img_list
        self.recon_directory = recon_info.recon_directory
        if self.volume_ground_truth is not None:
            self.birefringence_simulated = (
                self.volume_ground_truth.get_delta_n().detach()
            )
            mip_image = convert_volume_to_2d_mip(
                self.birefringence_simulated.unsqueeze(0)
            )
            self.birefringence_mip_sim = prepare_plot_mip(mip_image, plot=False)
        else:
            # Use the initial volume as a placeholder for plotting purposes
            self.birefringence_simulated = (
                self.volume_initial_guess.get_delta_n().detach()
            )
            mip_image = convert_volume_to_2d_mip(
                self.birefringence_simulated.unsqueeze(0)
            )
            self.birefringence_mip_sim = prepare_plot_mip(mip_image, plot=False)
        if self.intensity_imgs_meas:
            print("Intensity images were provided.")
        if output_dir is None:
            self.recon_directory = create_unique_directory("reconstructions")
        else:
            self.recon_directory = output_dir

        image_for_rays = None
        if omit_rays_based_on_pixels:
            image_for_rays = self.ret_img_meas
            print("Omitting rays based on pixels with zero retardance.")
        saved_ray_path = self.iteration_params.get("saved_ray_path", None)
        self.rays = self.setup_raytracer(
            image=image_for_rays, filepath=saved_ray_path, device=device
        )
        self.nerf_mode = self.iteration_params.get("nerf_mode", False)
        self.initialize_nerf_mode(use_nerf=self.nerf_mode)
        self.from_simulation = self.iteration_params.get("from_simulation", False)
        self.apply_volume_mask = apply_volume_mask
        self.mask = torch.ones(
            self.volume_initial_guess.Delta_n.shape[0], dtype=torch.bool, device=device
        )

        # Volume that will be updated after each iteration
        self.volume_pred = copy.deepcopy(self.volume_initial_guess)

        self.remove_large_arrs = self.iteration_params.get(
            "free_memory_by_del_large_arrays", False
        )
        if self.remove_large_arrs and self.apply_volume_mask:
            raise ValueError(
                "Cannot remove large arrays and apply mask to"
                "volume gradient at the same time."
            )
        self.two_optic_axis_components = self.iteration_params.get(
            "two_optic_axis_components", False
        )

        self.mla_rays_at_once = self.iteration_params.get("mla_rays_at_once", False)
        if self.mla_rays_at_once and not self.rays.MLA_volume_geometry_ready:
            self.rays.prepare_for_all_rays_at_once()
            if not self.from_simulation:
                radiometry_path = self.iteration_params.get("radiometry_path", None)
                if radiometry_path:
                    num_rays_og = self.rays.ray_valid_indices_all.shape[1]
                    radiometry = torch.tensor(recon_info.radiometry)
                    self.rays.filter_from_radiometry(radiometry)
                    num_rays = self.rays.ray_valid_indices_all.shape[1]
                    print(
                        f"Radiometry used for filtering rays from {num_rays_og} to {num_rays} rays."
                    )
                else:
                    print("No radiometry provided for filtering rays.")

        save_indices = False
        if save_indices:
            vox_indices_by_mla_idx = self.rays.vox_indices_by_mla_idx
            dict_save_dir = os.path.join(self.recon_directory, "config_parameters")
            if not os.path.exists(dict_save_dir):
                os.makedirs(dict_save_dir)
            dict_filename = "vox_indices_by_mla_idx.pkl"
            dict_save_path = os.path.join(dict_save_dir, dict_filename)
            with open(dict_save_path, "wb") as f:
                pickle.dump(vox_indices_by_mla_idx, f)
            print(f"Saving voxel indices by MLA index to {dict_save_path}")

        try:
            self.mask = self.rays.mask
        except AttributeError:
            self.voxel_mask_setup()

        save_rays = self.iteration_params.get("save_rays", False)
        # Ray saving should be done after self.rays.prepare_for_all_rays_at_once()
        if save_rays:
            rays_save_path = os.path.join(
                self.recon_directory, "config_parameters", "rays.pkl"
            )
            self.rays.save(rays_save_path)

        # Mask initial guess of volume
        self.apply_mask_to_volume(self.volume_pred)

        if self.remove_large_arrs:
            del self.birefringence_simulated
            gc.collect()

        datafidelity_method = self.iteration_params.get("datafidelity", "vector")
        first_word = datafidelity_method.split()[0]
        if first_word == "intensity":
            self.intensity_bool = True
            print("Using intensity images for data-fidelity term.")
        else:
            self.intensity_bool = False
            print("Using retardance and azimuth images for data-fidelity term.")

        # Lists to store the loss after each iteration
        self.loss_total_list = []
        self.loss_data_term_list = []
        self.loss_reg_term_list = []
        self.adjusted_lrs_list = []
        end_time = time.perf_counter()
        print(f"Reconstructor initialized in {end_time - start_time:.2f} seconds\n")

    def _initialize_volume(self):
        """
        Method to initialize volume if it's not provided.
        Here, we can return a default volume or use some initialization strategy.
        """
        # Placeholder for volume initialization
        default_volume = None
        return default_volume

    def _to_numpy(self, image):
        """Convert image to a numpy array, if it's not already."""
        if isinstance(image, torch.Tensor):
            return image.detach().cpu().numpy()
        elif isinstance(image, np.ndarray):
            return image
        else:
            raise TypeError("Image must be a PyTorch Tensor or a numpy array")

    def to_device(self, device):
        """
        Move all tensors to the specified device.
        """
        self.ret_img_meas = torch.from_numpy(self.ret_img_meas).to(device)
        self.azim_img_meas = torch.from_numpy(self.azim_img_meas).to(device)
        # self.volume_initial_guess = self.volume_initial_guess.to(device)
        if self.volume_ground_truth is not None:
            self.volume_ground_truth = self.volume_ground_truth.to(device)
        self.rays.to_device(device)
        self.mask = self.mask.to(device)
        self.volume_pred = self.volume_pred.to(device)

    def save_parameters(self, output_dir, volume_type):
        """In progress.
        Args:
            volume_type (dict): example volume_args.random_args
        """
        torch.save(
            {
                "optical_info": self.optical_info,
                "training_params": self.iteration_params,
                "volume_type": volume_type,
            },
            f"{output_dir}/parameters.pt",
        )

    @staticmethod
    def replace_nans(volume, ep):
        """Used in response to an error message."""
        with torch.no_grad():
            num_nan_vecs = torch.sum(torch.isnan(volume.optic_axis[0, :]))
            if num_nan_vecs > 0:
                replacement_vecs = torch.nn.functional.normalize(
                    torch.rand(3, int(num_nan_vecs)), p=2, dim=0
                )
                volume.optic_axis[:, torch.isnan(volume.optic_axis[0, :])] = (
                    replacement_vecs
                )
                if ep == 0:
                    print(
                        f"Replaced {num_nan_vecs} NaN optic axis vectors with random unit vectors."
                    )

    def setup_raytracer(self, image=None, filepath=None, device="cpu"):
        """Initialize Birefringent Raytracer."""
        if filepath:
            print(f"Loading rays from {filepath}")
            time0 = time.time()
            with open(filepath, "rb") as file:
                rays = pickle.load(file)
            # rays.MLA_volume_geometry_ready = True
            print(f"Loaded rays in {time.time() - time0:.0f} seconds")
        else:
            print(f"For raytracing, using computing device {device}")
            rays = BirefringentRaytraceLFM(
                backend=Reconstructor.backend, optical_info=self.optical_info
            )
            start_time = time.time()
            rays.compute_rays_geometry(filename=None, image=image)
            print(f"Raytracing time in seconds: {time.time() - start_time:.2f}")
        return rays

    def initialize_nerf_mode(self, use_nerf=True):
        self.rays.initialize_nerf_mode(use_nerf)

    def mask_outside_rays(self):
        """Mask out volume that is outside FOV of the microscope.
        Original shapes of the volume are preserved."""
        mask = self.rays.get_volume_reachable_region()
        with torch.no_grad():
            self.volume_pred.Delta_n[mask.view(-1) == 0] = 0
            # Masking the optic axis caused NaNs in the Jones Matrix. So, we don't mask it.
            # self.volume_pred.optic_axis[:, mask.view(-1)==0] = 0

    def crop_pred_volume_to_reachable_region(self):
        """Crop the predicted volume to the region that is reachable by the microscope.
        Note: This method modifies the volume_pred attribute. The voxel indices of the predetermined ray tracing are no longer valid.
        """
        mask = self.rays.get_volume_reachable_region()
        region_shape = get_region_of_ones_shape(mask).tolist()
        original_shape = self.optical_info["volume_shape"]
        self.optical_info["volume_shape"] = region_shape
        self.volume_pred.optical_info["volume_shape"] = region_shape
        birefringence = self.volume_pred.Delta_n
        optic_axis = self.volume_pred.optic_axis
        with torch.no_grad():
            cropped_birefringence = reshape_and_crop(
                birefringence, original_shape, region_shape
            )
            self.volume_pred.Delta_n = store_as_pytorch_parameter(
                cropped_birefringence, "scalar"
            )
            cropped_optic_axis = reshape_and_crop(
                optic_axis, [3, *original_shape], region_shape
            )
            self.volume_pred.optic_axis = store_as_pytorch_parameter(
                cropped_optic_axis, "vector"
            )

    def restrict_volume_to_reachable_region(self):
        """Restrict the volume to the region that is reachable by the microscope.
        This includes cropping the volume are creating a new ray geometry
        """
        self.crop_pred_volume_to_reachable_region()
        self.rays = self.setup_raytracer()

    def _turn_off_initial_volume_gradients(self):
        """Turn off the gradients for the initial volume guess."""
        self.volume_initial_guess.Delta_n.requires_grad = False
        self.volume_initial_guess.optic_axis.requires_grad = False

    def specify_variables_to_learn(self, learning_vars=None):
        """
        Specify which variables of the initial volume object should be considered for learning.
        This method updates the 'members_to_learn' attribute of the initial volume object, ensuring
        no duplicates are added.
        The variable names must be attributes of the BirefringentVolume class.
        Args:
            learning_vars (list): Variable names to be appended for learning.
                                    Defaults to ['Delta_n', 'optic_axis'].
        """
        volume = self.volume_pred
        if learning_vars is None:
            learning_vars = ["Delta_n", "optic_axis"]
        for var in learning_vars:
            if var not in volume.members_to_learn:
                volume.members_to_learn.append(var)

    def optimizer_setup(self, volume_estimation, training_params):
        """Setup optimizer."""
        trainable_parameters = volume_estimation.get_trainable_variables()
        trainable_vars_names = volume_estimation.get_names_of_trainable_variables()
        optimizer_type = training_params.get("optimizer", "Nadam")
        if optimizer_type == "LBFGS":
            parameters = trainable_parameters
        else:
            assert (
                len(trainable_parameters[0].shape) == 2
            ), "1st parameter should be the optic axis"
            assert (
                len(trainable_parameters[1].shape) == 1
            ), "2nd parameter should be the birefringence."
            # The learning rates specified are starting points for the optimizer.
            parameters = [
                {
                    "params": trainable_parameters[0],
                    "lr": training_params["lr_optic_axis"],
                    "name": trainable_vars_names[0],
                },
                {
                    "params": trainable_parameters[1],
                    "lr": training_params["lr_birefringence"],
                    "name": trainable_vars_names[1],
                },
            ]
        optimizers = {
            "Adam": lambda params: torch.optim.Adam(params),
            "SGD": lambda params: torch.optim.SGD(params, nesterov=True, momentum=0.7),
            "Adagrad": lambda params: torch.optim.Adagrad(params),
            "ASGD": lambda params: torch.optim.ASGD(params),
            "Nadam": lambda params: torch.optim.NAdam(params),
            "Adamax": lambda params: torch.optim.Adamax(params),
            "AdamW": lambda params: torch.optim.AdamW(params),
            "RMSprop": lambda params: torch.optim.RMSprop(params),
        }
        print(f"Using optimizer: {optimizer_type}")
        if optimizer_type == "LBFGS":
            raise ValueError(
                "LBFGS optimizer is not supported yet," + "because a closure is needed."
            )
        elif optimizer_type not in optimizers:
            raise ValueError(
                f"Unsupported optimizer type: {optimizer_type}."
                + f"Please choose from {list(optimizers.keys())}."
            )
        optimizer = optimizers[optimizer_type](parameters)
        return optimizer

    def voxel_mask_setup(self):
        """Extract volume voxel related information."""
        if self.rays.MLA_volume_geometry_ready:
            num_vox_in_volume = self.volume_pred.Delta_n.shape[0]
            print(
                f"Identifying the voxels that are reached by the rays out of the {num_vox_in_volume} voxels."
            )

            start_time = time.perf_counter()
            filtered_voxels = filter_voxels_using_retardance(
                self.rays.vox_indices_ml_shifted_all,
                self.rays.ray_valid_indices_all,
                self.ret_img_meas,
            )

            mask = torch.zeros(num_vox_in_volume, dtype=torch.bool)
            mask[filtered_voxels] = True
            self.mask = mask
            self.rays.mask = mask  # Created as a rays arribute for saving purposes

            end_time = time.perf_counter()
            print(f"Voxel mask created in {end_time - start_time:.2f} seconds")
        else:
            try:
                vox_indices_path = self.iteration_params["vox_indices_by_mla_idx_path"]
                if not vox_indices_path:
                    raise ValueError("Vox indices path is empty.")
                start_time = time.perf_counter()
                with open(vox_indices_path, "rb") as f:
                    vox_indices_by_mla_idx = pickle.load(f)
                end_time = time.perf_counter()
                print(
                    f"Voxel indices by MLA index loaded in {end_time - start_time:.0f} seconds from {vox_indices_path}"
                )
                self.rays.vox_indices_by_mla_idx = vox_indices_by_mla_idx
            except (KeyError, FileNotFoundError, ValueError) as e:
                print(f"KeyError, FileNotFoundError, or ValueError occured: {e}")
                self.rays.store_shifted_vox_indices()
                vox_indices_by_mla_idx = self.rays.vox_indices_by_mla_idx
                dict_save_dir = os.path.join(self.recon_directory, "config_parameters")
                if not os.path.exists(dict_save_dir):
                    os.makedirs(dict_save_dir)
                dict_filename = "vox_indices_by_mla_idx.pkl"
                dict_save_path = os.path.join(dict_save_dir, dict_filename)
                with open(dict_save_path, "wb") as f:
                    pickle.dump(vox_indices_by_mla_idx, f)
                print(f"Saving voxel indices by MLA index to {dict_save_path}")
            if DEBUG:
                check_for_negative_values_dict(vox_indices_by_mla_idx)

            print("Collecting the set of voxels that are reached by the rays.")
            start_time = time.perf_counter()
            ### Examining values in terms of sets
            # vox_set = extract_numbers_from_dict_of_lists(vox_indices_by_mla_idx)
            # vox_sets_by_mla_idx = transform_dict_list_to_set(vox_indices_by_mla_idx)
            vox_set = set(self.rays.identify_voxels_at_least_one_nonzero_ret())
            # Excluding voxels that are a part of multiple zero-retardance rays
            if self.from_simulation:
                vox_list_excluding = self.rays.identify_voxels_repeated_zero_ret()
            else:
                vox_list_excluding = self.rays.identify_voxels_zero_ret_lenslet()
            if DEBUG:
                check_for_negative_values(vox_list_excluding)
            filtered_vox_list = list(vox_set - set(vox_list_excluding))
            sorted_vox_list = sorted(filtered_vox_list)
            print(
                f"Masking out voxels except for {len(sorted_vox_list)} voxels. "
                + f"First, at most, 20 voxels are {sorted_vox_list[:20]}"
            )
            vox_set_tensor = torch.tensor(sorted_vox_list, dtype=torch.long)
            Delta_n = self.volume_pred.Delta_n
            mask = torch.zeros(Delta_n.shape[0], dtype=torch.bool)
            mask[vox_set_tensor] = True
            self.mask = mask
            self.rays.mask = mask  # Created as a rays arribute for saving purposes
            end_time = time.perf_counter()
            print(f"Voxel mask created in {end_time - start_time:.2f} seconds")
        return

    def apply_mask_to_volume(self, volume: BirefringentVolume):
        volume.Delta_n = torch.nn.Parameter(
            volume.Delta_n * self.mask.to(volume.Delta_n.device)
        )
        return

    def _compute_loss(self, images_predicted: list):
        """
        Compute the loss for the current iteration after the forward
        model is applied.

        Args:
            images_predicted (list): A list of images that are output
                from the forward model, could be ret/azim images or
                intensity images.

        Returns:
            loss (torch.Tensor): The total loss.

        Note: If ep is a class attibrute, then the loss function can
        depend on the current epoch.
        """
        vol_pred = self.volume_pred
        params = self.iteration_params
        retardance_meas = self.ret_img_meas
        azimuth_meas = self.azim_img_meas
        intensity_imgs_meas = self.intensity_imgs_meas

        if not torch.is_tensor(retardance_meas):
            retardance_meas = torch.tensor(retardance_meas)
        if not torch.is_tensor(azimuth_meas):
            azimuth_meas = torch.tensor(azimuth_meas)
        if intensity_imgs_meas is not None:
            for i, img in enumerate(intensity_imgs_meas):
                if not torch.is_tensor(img):
                    intensity_imgs_meas[i] = torch.tensor(img)

        # TODO: move these initializations so that they are only done once
        LossFcn = PolarimetricLossFunction(params=params)
        LossFcn.set_retardance_target(retardance_meas)
        LossFcn.set_orientation_target(azimuth_meas)
        LossFcn.set_intensity_list_target(intensity_imgs_meas)
        LossFcn.mask = self.mask
        data_term = LossFcn.compute_datafidelity_term(
            LossFcn.datafidelity, images_predicted
        )

        # Compute regularization term
        if isinstance(params["regularization_weight"], list):
            params["regularization_weight"] = params["regularization_weight"][0]
        reg_loss, reg_term_values = LossFcn.compute_regularization_term(vol_pred)
        regularization_term = params["regularization_weight"] * reg_loss
        self.reg_term_values = [reg.item() for reg in reg_term_values]

        # Total loss
        loss = data_term + regularization_term

        return loss, data_term, regularization_term

    def keep_optic_axis_on_sphere(self, volume):
        """Method to keep the optic axis on the unit sphere."""
        if volume.indices_active is not None:
            optic_axis = volume.optic_axis_active
        else:
            optic_axis = volume.optic_axis
        optic_axis = stay_on_sphere(optic_axis)
        return

    def fill_optaxis_component(self, volume):
        """Method to fill the axial component of the optic axis
        with the square root of the remaining components.

        Also, here the updated optic axis is stored in the volume object.
        """
        fill_vector_based_on_nonaxial(
            volume.optic_axis_active, volume.optic_axis_planar
        )
        return

    # @profile # to see the memory breakdown of the function
    def one_iteration(self, optimizer, volume_estimation, scheduler=None):
        if not self.apply_volume_mask:
            optimizer.zero_grad()
        else:
            # improving memory usage by setting gradients to None
            optimizer.zero_grad(set_to_none=True)
        # Apply forward model and compute loss
        img_list = self.rays.ray_trace_through_volume(
            volume_estimation,
            intensity=self.intensity_bool,
            all_rays_at_once=self.mla_rays_at_once,
        )
        # In case the entire volume is needed for the loss computation:
        total_vol_needed = False
        if total_vol_needed and self.volume_pred.indices_active is not None:
            with torch.no_grad():
                self.volume_pred.Delta_n[self.volume_pred.indices_active] = (
                    self.volume_pred.birefringence_active
                )
                self.volume_pred.optic_axis[:, self.volume_pred.indices_active] = (
                    self.volume_pred.optic_axis_active
                )
        if self.nerf_mode:
            # Update Delta_n before loss is computed so the the mask regularization is applied
            vol_shape = self.optical_info["volume_shape"]
            predicted_properties = predict_voxel_properties(
                self.rays.inr_model, vol_shape, enable_grad=True
            )
            Delta_n = predicted_properties[..., 0]
            # # Gradients are lost when setting Delta_n as a torch nn parameter
            # self.volume_pred.Delta_n = torch.nn.Parameter(Delta_n.flatten())
            self.volume_pred.birefringence = Delta_n

        loss, data_term, regularization_term = self._compute_loss(img_list)
        if self.rays.verbose:
            tqdm.write(f"Computed the loss: {loss.item():.5}")

        # Verify the gradients before and after the backward pass
        if PRINT_GRADIENTS:
            print("\nBefore backward pass:")
            self.print_grad_info(volume_estimation)

        loss.backward()

        if PRINT_GRADIENTS:
            print("\nAfter backward pass:")
            self.print_grad_info(volume_estimation)

        if CLIP_GRADIENT_NORM:
            self.clip_gradient_norms(optimizer, volume_estimation)

        # Apply voxel-specific mask
        if self.apply_volume_mask:
            with torch.no_grad():
                self.volume_pred.Delta_n.grad *= self.mask

        optimizer.step()
        scheduler.step(loss)
        adj_lrs_dict = calculate_adjusted_lr(optimizer)
        if self.nerf_mode:
            adjusted_lrs = [0]
        else:
            adjusted_lrs = [val.item() for val in adj_lrs_dict.values()]

        if PRINT_GRADIENTS:
            print_moments(optimizer)

        # Keep the optic axis on the unit sphere
        if self.two_optic_axis_components:
            self.fill_optaxis_component(volume_estimation)
        self.keep_optic_axis_on_sphere(volume_estimation)

        if self.ep % 50 == 0 and False:
            tqdm.write(f"Iteration {self.ep} first 5 values:")
            tqdm.write(
                f"birefringence: {volume_estimation.birefringence_active[:5].detach().cpu().numpy()}"
            )
            tqdm.write(
                f"optic axis: {volume_estimation.optic_axis_active[:, :5].detach().cpu().numpy()}"
            )
        # TODO: fix so that measured images do not need to be placeholder for the predicted images
        if self.intensity_bool:
            regenerate = False
            if regenerate:
                # Alternatively, the ray tracing can be done again without intensity boolean.
                with torch.no_grad():
                    [ret_image_current, azim_image_current] = (
                        self.rays.ray_trace_through_volume(volume_estimation)
                    )
                self.store_results(
                    ret_image_current,
                    azim_image_current,
                    volume_estimation,
                    loss,
                    data_term,
                    regularization_term,
                    adjusted_lrs,
                )
            else:
                [self.ret_img_pred, self.azim_img_pred] = (
                    self.ret_img_meas,
                    self.azim_img_meas,
                )
                self.volume_pred = volume_estimation
                self.loss_total_list.append(loss.item())
                self.loss_data_term_list.append(data_term.item())
                self.loss_reg_term_list.append(regularization_term.item())
                self.adjusted_lrs_list.append(adjusted_lrs)
        else:
            [ret_image_current, azim_image_current] = img_list
            self.store_results(
                ret_image_current,
                azim_image_current,
                volume_estimation,
                loss,
                data_term,
                regularization_term,
                adjusted_lrs,
            )
        return

    def print_grad_info(self, volume_estimation):
        if False:
            print(
                "Delta_n requires_grad:",
                volume_estimation.Delta_n.requires_grad,
                "birefringence_active requires_grad:",
                volume_estimation.birefringence_active.requires_grad,
            )
            if volume_estimation.Delta_n.grad is not None:
                print(
                    "Gradient for Delta_n (up to 10 values):",
                    volume_estimation.Delta_n.grad[:10],
                )
            else:
                print("Gradient for Delta_n is None")
        if volume_estimation.birefringence_active.grad is not None:
            print(
                "Gradient for birefringence_active (up to 10 values):",
                volume_estimation.birefringence_active.grad[:10],
            )
        else:
            print("Gradient for birefringence_active is None")

    def store_results(
        self,
        ret_image_current,
        azim_image_current,
        volume_estimation,
        loss,
        data_term,
        regularization_term,
        adjusted_lrs,
    ):
        self.ret_img_pred = ret_image_current.detach().cpu().numpy()
        self.azim_img_pred = azim_image_current.detach().cpu().numpy()
        self.volume_pred = volume_estimation
        self.loss_total_list.append(loss.item())
        self.loss_data_term_list.append(data_term.item())
        self.loss_reg_term_list.append(regularization_term.item())
        self.adjusted_lrs_list.append(adjusted_lrs)

    def visualize_and_save(self, ep, fig, output_dir):
        volume_estimation = self.volume_pred
        if self.remove_large_arrs:
            vol_shape = self.optical_info["volume_shape"]
            temp_bir = torch.zeros(vol_shape).flatten()
            device = volume_estimation.birefringence_active.device
            volume_estimation.Delta_n = torch.nn.Parameter(
                temp_bir, requires_grad=False
            ).to(device)
        if self.volume_pred.indices_active is not None:
            with torch.no_grad():
                volume_estimation.Delta_n[volume_estimation.indices_active] = (
                    volume_estimation.birefringence_active
                )

        save_freq = self.iteration_params.get("save_freq", 5)
        # TODO: only update every 1 epoch if plotting is live
        if ep % 1 == 0:
            # plt.clf()
            if self.nerf_mode:
                vol_shape = self.optical_info["volume_shape"]
                predicted_properties = predict_voxel_properties(
                    self.rays.inr_model, vol_shape
                )
                Delta_n = predicted_properties[..., 0]
                volume_estimation.Delta_n = torch.nn.Parameter(Delta_n.flatten())
                # TODO: see if mask should be applied here
                volume_estimation.Delta_n = torch.nn.Parameter(
                    volume_estimation.Delta_n * self.mask
                )
                Delta_n = volume_estimation.get_delta_n().detach().unsqueeze(0)
            else:
                Delta_n = volume_estimation.get_delta_n().detach().unsqueeze(0)
            mip_image = convert_volume_to_2d_mip(Delta_n)
            mip_image_np = prepare_plot_mip(mip_image, plot=False)
            plot_iteration_update_gridspec(
                self.birefringence_mip_sim,
                self.ret_img_meas,
                self.azim_img_meas,
                mip_image_np,
                self.ret_img_pred,
                self.azim_img_pred,
                self.loss_total_list,
                self.loss_data_term_list,
                self.loss_reg_term_list,
                figure=fig,
            )
            fig.canvas.draw()
            fig.canvas.flush_events()
            time.sleep(0.1)
            self.save_loss_lists_to_csv()
            self._save_regularization_terms_to_csv(ep)
            if ep % save_freq == 0:
                filename = f"optim_ep_{'{:04d}'.format(ep)}.pdf"
                plt.savefig(os.path.join(output_dir, filename))
            time.sleep(0.1)
        if ep % save_freq == 0:
            if self.remove_large_arrs:
                vol_size_flat = volume_estimation.Delta_n.size(0)
                device = volume_estimation.optic_axis_active.device
                volume_estimation.optic_axis = torch.nn.Parameter(
                    torch.zeros(3, vol_size_flat), requires_grad=False
                ).to(device)
            if self.nerf_mode:
                optic_axis_flat = predicted_properties.view(
                    -1, predicted_properties.shape[-1]
                )[..., 1:]
                if predicted_properties.shape[-1] == 3:
                    optic_axis_flat = spherical_to_unit_vector_torch(optic_axis_flat)
                volume_estimation.optic_axis = torch.nn.Parameter(
                    optic_axis_flat.permute(1, 0)
                )
                nerf_model_path = os.path.join(output_dir, f"nerf_model_{ep}.pth")
                self.rays.save_nerf_model(nerf_model_path)
            else:
                if self.volume_pred.indices_active is not None:
                    with torch.no_grad():
                        volume_estimation.optic_axis[
                            :, volume_estimation.indices_active
                        ] = volume_estimation.optic_axis_active
            my_description = "Volume estimation after " + str(ep) + " iterations."
            volume_estimation.save_as_file(
                os.path.join(output_dir, f"volume_ep_{'{:04d}'.format(ep)}.h5"),
                description=my_description,
            )
            if self.remove_large_arrs:
                del volume_estimation.optic_axis
                gc.collect()
        if self.remove_large_arrs:
            del volume_estimation.Delta_n
            gc.collect()
        return

    def __visualize_and_update_streamlit(
        self, progress_bar, ep, n_epochs, recon_img_plot, my_loss
    ):
        import pandas as pd

        percent_complete = int(ep / n_epochs * 100)
        progress_bar.progress(percent_complete + 1)
        if ep % 2 == 0:
            plt.close()
            recon_img_fig = plot_retardance_orientation(
                self.ret_img_pred, self.azim_img_pred, "hsv"
            )
            recon_img_plot.pyplot(recon_img_fig)
            df_loss = pd.DataFrame(
                {
                    "Total loss": self.loss_total_list,
                    "Data fidelity": self.loss_data_term_list,
                    "Regularization": self.loss_reg_term_list,
                }
            )
            my_loss.line_chart(df_loss)

    def save_loss_lists_to_csv(self):
        """Save the loss lists to a csv file.

        Class instance attributes accessed:
        - self.recon_directory
        - self.loss_total_list
        - self.loss_data_term_list
        - self.loss_reg_term_list
        - self.adjusted_lrs_list
        """
        filename = "loss.csv"
        filepath = os.path.join(self.recon_directory, filename)

        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    "Total Loss",
                    "Data Term Loss",
                    "Regularization Term Loss",
                    "Optic Axis Learning Rate",
                    "Birefringence Learning Rate",
                ]
            )
            zipped_lists = zip(
                self.loss_total_list,
                self.loss_data_term_list,
                self.loss_reg_term_list,
                self.adjusted_lrs_list,
            )
            if self.nerf_mode:
                for total, data_term, reg_term, lr in zipped_lists:
                    writer.writerow([total, data_term, reg_term, lr])
            else:
                for total, data_term, reg_term, (optax_lr, bir_lr) in zipped_lists:
                    writer.writerow([total, data_term, reg_term, optax_lr, bir_lr])

    def _create_regularization_terms_csv(self):
        """Create a csv file to store the regularization terms."""
        filename = "regularization_terms.csv"
        filepath = os.path.join(self.recon_directory, filename)
        reg_fcns = self.iteration_params["regularization_fcns"]
        fcn_names = [sublist[0] for sublist in reg_fcns]
        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["ep", *fcn_names])

    def _save_regularization_terms_to_csv(self, ep):
        """Save the regularization terms to a csv file."""
        filename = "regularization_terms.csv"
        filepath = os.path.join(self.recon_directory, filename)
        with open(filepath, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([ep, *self.reg_term_values])

    def clip_gradient_norms(self, model, verbose=False):
        # Gradient clipping
        max_norm = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        if verbose:
            # Calculate the total norm of the gradients
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm**0.5
            if total_norm > max_norm:
                print(
                    f"Epoch {self.ep}: Gradients clipped (total_norm: {total_norm:.2f})"
                )

    def create_parameters_from_mask(self, volume, mask):
        """Create volume attributes from the volume prperties and
        the mask. These attributes are intended for optimization."""
        active_indices = torch.where(mask)[0]
        device = volume.Delta_n.device
        volume.indices_active = active_indices.to(device)
        max_index = mask.size()[0]
        idx_tensor = torch.full((max_index + 1,), -1, dtype=torch.long).to(device)
        positions = torch.arange(len(active_indices), dtype=torch.long).to(device)
        idx_tensor[active_indices] = positions
        volume.active_idx2spatial_idx_tensor = idx_tensor
        if self.two_optic_axis_components:
            volume.optic_axis.requires_grad = False
            volume.optic_axis_active = volume.optic_axis[:, active_indices]
            volume.optic_axis_planar = torch.nn.Parameter(
                volume.optic_axis_active[1:, :]
            )
        else:
            volume.optic_axis_active = torch.nn.Parameter(
                volume.optic_axis[:, active_indices]
            )
        volume.birefringence_active = torch.nn.Parameter(volume.Delta_n[active_indices])

    def prepare_volume_for_recon(self, volume):
        if self.two_optic_axis_components:
            self.fill_optaxis_component(volume)
        self.keep_optic_axis_on_sphere(volume)
        check_for_inf_or_nan(volume.birefringence_active)
        check_for_inf_or_nan(volume.optic_axis_active)

    def reconstruct(
        self,
        use_streamlit=False,
        plot_live=False,
        all_prop_elements=False,
        log_file=None,
    ):
        """
        Method to perform the actual reconstruction based on the provided parameters.
        """
        print(f"Beginning reconstruction iterations...")
        # Turn off the gradients for the initial volume guess
        self._turn_off_initial_volume_gradients()

        # Specify variables to learn
        if all_prop_elements:
            param_list = ["Delta_n", "optic_axis"]
        else:
            self.create_parameters_from_mask(self.volume_pred, self.mask)
            if self.two_optic_axis_components:
                param_list = ["birefringence_active", "optic_axis_planar"]
            else:
                param_list = ["birefringence_active", "optic_axis_active"]
            if self.remove_large_arrs:
                del self.volume_pred.Delta_n
                del self.volume_pred.optic_axis
                gc.collect()
            else:
                self.volume_pred.Delta_n.detach()
                self.volume_pred.Delta_n.requires_grad = False
                self.volume_pred.optic_axis.detach()
                self.volume_pred.optic_axis.requires_grad = False
        self.specify_variables_to_learn(param_list)

        if self.nerf_mode:
            optimizer = setup_optimizer_nerf(self.rays, self.iteration_params)
        else:
            optimizer = self.optimizer_setup(self.volume_pred, self.iteration_params)
            optax_betas = self.iteration_params.get("optax_betas", (0.9, 0.999))
            bir_betas = self.iteration_params.get("bir_betas", (0.9, 0.999))
            optimizer.param_groups[0]["betas"] = tuple(optax_betas)
            optimizer.param_groups[1]["betas"] = tuple(bir_betas)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            threshold=1e-4,
            threshold_mode="rel",
            cooldown=0,
            min_lr=1e-6,
            eps=1e-8,
        )
        figure = setup_visualization(
            window_title=self.recon_directory, plot_live=plot_live
        )
        self._create_regularization_terms_csv()

        n_epochs = self.iteration_params["n_epochs"]
        if use_streamlit:
            import streamlit as st

            st.write("Working on these ", n_epochs, "iterations...")
            my_recon_img_plot = st.empty()
            my_loss = st.empty()
            my_plot = st.empty()  # set up a place holder for the plot
            my_3D_plot = st.empty()  # set up a place holder for the 3D plot
            progress_bar = st.progress(0)

        self.prepare_volume_for_recon(self.volume_pred)
        initial_lr_0 = optimizer.param_groups[0]["lr"]
        if self.nerf_mode:
            initial_lr_1 = optimizer.param_groups[0]["lr"]
        else:
            initial_lr_1 = optimizer.param_groups[1]["lr"]
        # Parameters for learning rate warmup

        warmup_epochs = 10
        warmup_start_proportion = 0.1
        # Iterations
        for ep in tqdm(range(1, n_epochs + 1), "Minimizing"):
            self.ep = ep
            # Learning rate warmup
            if ep < warmup_epochs:
                lr_0 = initial_lr_0 * (
                    warmup_start_proportion
                    + (1 - warmup_start_proportion) * (ep / warmup_epochs)
                )
                lr_1 = initial_lr_1 * (
                    warmup_start_proportion
                    + (1 - warmup_start_proportion) * (ep / warmup_epochs)
                )
                optimizer.param_groups[0]["lr"] = lr_0
                if not self.nerf_mode:
                    optimizer.param_groups[1]["lr"] = lr_1
            else:
                current_lr_0 = scheduler.optimizer.param_groups[0]["lr"]
                if self.nerf_mode:
                    current_lr_1 = lr_0
                else:
                    current_lr_1 = scheduler.optimizer.param_groups[1]["lr"]
                if lr_0 != current_lr_0 or lr_1 != current_lr_1:
                    print(
                        f"Learning rates at iteration {ep - 1}: {lr_0:.2e}, {lr_1:.2e}"
                    )
                    print(f"Learning rates changed at epoch {ep}")
                    print(
                        f"Learning rates at iteration {ep}: {current_lr_0:.2e}, {current_lr_1:.2e}"
                    )
                else:
                    pass
                lr_0 = current_lr_0
                lr_1 = current_lr_1
            self.one_iteration(optimizer, self.volume_pred, scheduler=scheduler)
            if ep == 1 and PRINT_TIMING_INFO:
                self.rays.print_timing_info()
            if ep % 20 == 0 and self.intensity_bool:
                with torch.no_grad():
                    [ret_image_current, azim_image_current] = (
                        self.rays.ray_trace_through_volume(self.volume_pred)
                    )
                self.ret_img_pred = ret_image_current.detach().cpu().numpy()
                self.azim_img_pred = azim_image_current.detach().cpu().numpy()
            sys.stdout.flush()

            azim_damp_mask = self._to_numpy(self.ret_img_meas / self.ret_img_meas.max())
            self.azim_img_pred[azim_damp_mask == 0] = 0
            if use_streamlit:
                self.__visualize_and_update_streamlit(
                    progress_bar, ep, n_epochs, my_recon_img_plot, my_loss
                )
            self.visualize_and_save(ep, figure, self.recon_directory)

        self.save_loss_lists_to_csv()
        if self.remove_large_arrs:
            vol_shape = self.optical_info["volume_shape"]
            temp_bir = torch.zeros(vol_shape).flatten()
            device = self.volume_pred.birefringence_active.device
            self.volume_pred.Delta_n = torch.nn.Parameter(
                temp_bir, requires_grad=False
            ).to(device)
            self.volume_pred.Delta_n[self.volume_pred.indices_active] = (
                self.volume_pred.birefringence_active
            )
            vol_size_flat = self.volume_pred.Delta_n.size(0)
            self.volume_pred.optic_axis = torch.nn.Parameter(
                torch.zeros(3, vol_size_flat), requires_grad=False
            ).to(device)
            self.volume_pred.optic_axis[:, self.volume_pred.indices_active] = (
                self.volume_pred.optic_axis_active
            )

        my_description = "Volume estimation after " + str(ep) + " iterations."
        vol_save_path = os.path.join(
            self.recon_directory, f"volume_ep_{'{:04d}'.format(ep)}.h5"
        )
        self.volume_pred.save_as_file(vol_save_path, description=my_description)
        print("Saved the final volume estimation to", vol_save_path)
        plt.savefig(os.path.join(self.recon_directory, "optim_final.pdf"))
        plt.close()

        if self.nerf_mode:
            nerf_model_path = os.path.join(self.recon_directory, "nerf_model.pth")
            self.rays.save_nerf_model(nerf_model_path) 
