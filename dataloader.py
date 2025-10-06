import os
import zarr
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import scipy.ndimage

class ZarrSegmentationDataset:
    """
    A simple dataloader for 3D EM segmentation datasets stored in Zarr format.

    This dataset class scans a root directory to find pairs of raw EM volumes
    and their corresponding labeled segmentation crops. It assumes a specific
    directory structure as outlined in the user's request.
    """

    def __init__(self, root_dir, raw_scale='s0', labels_scale='s0'):
        """
        Initializes the dataset by scanning for valid data samples.

        Args:
            root_dir (str): The path to the root directory containing the Zarr datasets.
            raw_scale (str, optional): The scale level to use for the raw data. Defaults to 's0'.
            labels_scale (str, optional): The scale level to use for the labels. Defaults to 's0'.
        """
        if scipy is None:
            raise ImportError("This dataset requires the 'scipy' library. Please install it.")
        self.root_dir = root_dir
        self.raw_scale = raw_scale
        self.labels_scale = labels_scale
        self.samples = self._find_samples()

        if not self.samples:
            print(f"Warning: No valid samples found in {self.root_dir}. Please check the directory structure and file paths.")

    def _find_samples(self):
        """
        Scans the root directory to find all (raw_volume, label_crop) pairs.

        Returns:
            list: A list of dictionaries, where each dictionary contains paths and metadata for a single sample.
        """
        samples = []
        # Find all top-level zarr directories
        zarr_paths = glob(os.path.join(self.root_dir, '*/*.zarr'))

        for zarr_path in zarr_paths:
            try:
                zarr_root = zarr.open(zarr_path, mode='r')
            except Exception as e:
                print(f"Could not open {zarr_path}, skipping. Error: {e}")
                continue

            # Iterate through reconstruction groups (e.g., recon-1)
            for recon_name in zarr_root.keys():
                if not recon_name.startswith('recon-'):
                    continue

                raw_path_str = os.path.join(recon_name, 'em', 'fibsem-uint8', self.raw_scale)
                labels_base_path_str = os.path.join(recon_name, 'labels', 'groundtruth')

                if raw_path_str not in zarr_root or labels_base_path_str not in zarr_root:
                    continue
                
                # Find all available crops for this reconstruction
                groundtruth_group = zarr_root[labels_base_path_str]
                for crop_name in groundtruth_group.keys():
                    if not crop_name.startswith('crop'):
                        continue
                    
                    # We will use the 'all' mask which contains all label classes
                    label_path_str = os.path.join(labels_base_path_str, crop_name, 'all', self.labels_scale)
                    
                    if label_path_str in zarr_root:
                        samples.append({
                            'zarr_path': zarr_path,
                            'raw_path': raw_path_str,
                            'label_path': label_path_str
                        })
                        
        return samples

    def __len__(self):
        """Returns the total number of samples (label crops) found."""
        return len(self.samples)
    
    def _parse_ome_ngff_metadata(self, attrs, scale_level_name):
        """Helper function to parse scale and translation from OME-NGFF metadata."""
        try:
            multiscales = attrs['multiscales'][0]
            datasets = multiscales['datasets']
            scale_metadata = next((d for d in datasets if d['path'] == scale_level_name), None)
            
            if scale_metadata:
                transformations = scale_metadata['coordinateTransformations']
                scale_transform = next((t for t in transformations if t['type'] == 'scale'), None)
                translation_transform = next((t for t in transformations if t['type'] == 'translation'), None)
                
                scale = scale_transform['scale'] if scale_transform else [1.0, 1.0, 1.0]
                translation = translation_transform['translation'] if translation_transform else [0.0, 0.0, 0.0]
                
                return scale, translation
        except (KeyError, IndexError, StopIteration):
            pass # We will handle the error outside this function
        
        return None, None


    def __getitem__(self, idx):
        """
        Fetches a single raw crop and its corresponding segmentation mask, ensuring
        they correspond to the same physical volume and have the same voxel dimensions.
        """
        sample_info = self.samples[idx]
        zarr_path = sample_info['zarr_path']
        
        zarr_root = zarr.open(zarr_path, mode='r')

        raw_array = zarr_root[sample_info['raw_path']]
        label_array = zarr_root[sample_info['label_path']]

        # --- Parse Metadata for Both Raw and Label Volumes ---
        raw_attrs_group_path = os.path.dirname(sample_info['raw_path'])
        raw_attrs = zarr_root[raw_attrs_group_path].attrs.asdict()
        raw_scale_name = os.path.basename(sample_info['raw_path'])
        raw_scale, raw_translation = self._parse_ome_ngff_metadata(raw_attrs, raw_scale_name)
        if raw_scale is None:
            print(f"Warning: Could not parse metadata for raw volume {raw_attrs_group_path}. Assuming scale=[1,1,1] and translation=[0,0,0].")
            raw_scale, raw_translation = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]

        label_attrs_group_path = os.path.dirname(sample_info['label_path'])
        label_attrs = zarr_root[label_attrs_group_path].attrs.asdict()
        label_scale_name = os.path.basename(sample_info['label_path'])
        label_scale, label_translation = self._parse_ome_ngff_metadata(label_attrs, label_scale_name)
        if label_scale is None:
             raise ValueError(f"Could not parse required OME-NGFF metadata from {label_attrs_group_path}")

        print(f"Raw scale: {raw_scale}, Raw translation: {raw_translation}")
        print(f"Label scale: {label_scale}, Label translation: {label_translation}")

        # --- Calculate Crop Parameters in the Raw Volume's Voxel Space ---
        # 1. Physical start of label crop relative to the raw volume's physical origin
        relative_translation = [lt - rt for lt, rt in zip(label_translation, raw_translation)]

        # 2. Voxel start of the crop inside the raw array
        start_voxels_raw = [int(round(t / s)) for t, s in zip(relative_translation, raw_scale)]

        # 3. Physical size of the entire label volume
        label_shape_voxels = label_array.shape
        label_physical_size = [sh * sc for sh, sc in zip(label_shape_voxels, label_scale)]

        # 4. Required shape of the crop in raw-volume-voxels to match the physical size
        crop_shape_raw = [int(round(ps / rs)) for ps, rs in zip(label_physical_size, raw_scale)]

        # --- Extract Raw Crop ---
        z_start, y_start, x_start = start_voxels_raw
        slicing = (
            slice(z_start, z_start + crop_shape_raw[0]),
            slice(y_start, y_start + crop_shape_raw[1]),
            slice(x_start, x_start + crop_shape_raw[2]),
        )
        raw_crop = raw_array[slicing]

        # --- Resample Label Mask to Match the Raw Crop ---
        label_data = label_array[:]
        
        # 1. Calculate the zoom factor needed to go from label resolution to raw resolution
        zoom_factor = [ls / rs for ls, rs in zip(label_scale, raw_scale)]

        # 2. Resample using nearest-neighbor interpolation (order=0) to preserve labels
        resampled_label_mask = scipy.ndimage.zoom(label_data, zoom_factor, order=0)
        
        # 3. Ensure shapes match exactly after rounding by padding or cropping.
        final_shape = raw_crop.shape
        final_label_mask = np.zeros(final_shape, dtype=resampled_label_mask.dtype)
        
        slicing_for_copy = tuple(slice(0, min(fs, cs)) for fs, cs in zip(final_shape, resampled_label_mask.shape))
        
        final_label_mask[slicing_for_copy] = resampled_label_mask[slicing_for_copy]

        return raw_crop, final_label_mask

if __name__ == '__main__':
    
    DATA_DIR = '/lustre/gale/stf218/scratch/emin/cellmap-segmentation-challenge_old/data'

    print("\nInitializing ZarrSegmentationDataset...")
    # Initialize the dataset with the root directory containing the datasets.
    dataset = ZarrSegmentationDataset(root_dir=DATA_DIR)

    print(f"Found {len(dataset)} samples.")

    if len(dataset) > 0:
        print("\nFetching the first sample...")
        # Get the first sample
        raw_image_crop, label_mask_crop = dataset[0]

        print(f"Raw image crop shape: {raw_image_crop.shape}")
        print(f"Label mask crop shape: {label_mask_crop.shape}")
        print(f"Unique labels: {np.unique(label_mask_crop)}")

        # Verify the shapes match
        assert raw_image_crop.shape == label_mask_crop.shape
        print("\nSuccessfully loaded a sample and shapes match!")

        # --- Visualization Test ---
        print("\nRunning visualization test...")
            
        num_slices_to_show = 4  # Ensure the crop has enough depth to display
        fig, axes = plt.subplots(num_slices_to_show, 2, figsize=(10, 2 * num_slices_to_show))
        fig.suptitle('Visual Correspondence Check', fontsize=16)

        # Select evenly spaced slice indices from the z-axis
        slice_indices = np.linspace(0, raw_image_crop.shape[0] - 1, num_slices_to_show, dtype=int)

        for i, slice_idx in enumerate(slice_indices):
            # Plot raw image slice
            axes[i, 0].imshow(raw_image_crop[slice_idx, :, :], cmap='gray')
            axes[i, 0].set_title(f'Raw Slice (Z={slice_idx})')
            axes[i, 0].axis('off')

            # Plot label mask slice
            axes[i, 1].imshow(label_mask_crop[slice_idx, :, :])
            axes[i, 1].set_title(f'Label Mask (Z={slice_idx})')
            axes[i, 1].axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig("raw_labeled_crops.jpeg", bbox_inches='tight')
        print("Figure successfully saved.")