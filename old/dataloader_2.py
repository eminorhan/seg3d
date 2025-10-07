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
    and their corresponding labeled segmentation crops. It returns fixed-size
    crops suitable for training deep learning models.
    """

    def __init__(self, root_dir, raw_scale='s0', labels_scale='s0', output_size=(512, 512, 512)):
        """
        Initializes the dataset by scanning for valid data samples.

        Args:
            root_dir (str): The path to the root directory containing the Zarr datasets.
            raw_scale (str, optional): The default highest-resolution scale for raw data. Defaults to 's0'.
            labels_scale (str, optional): The scale level to use for the labels. Defaults to 's0'.
            output_size (tuple, optional): The desired (Z, Y, X) output size of the raw and label crops. Defaults to (512, 512, 512).
        """
        if scipy is None:
            raise ImportError("This dataset requires the 'scipy' library. Please install it.")
        self.root_dir = root_dir
        self.raw_scale = raw_scale
        self.labels_scale = labels_scale
        self.output_size = output_size
        self.samples = self._find_samples()

        if not self.samples:
            print(f"Warning: No valid samples found in {self.root_dir}. Please check the directory structure and file paths.")

    def _find_samples(self):
        """
        Scans the root directory to find all (raw_volume_group, label_crop) pairs.

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

                raw_group_path_str = os.path.join(recon_name, 'em', 'fibsem-uint8')
                labels_base_path_str = os.path.join(recon_name, 'labels', 'groundtruth')

                if raw_group_path_str not in zarr_root or labels_base_path_str not in zarr_root:
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
                            'raw_path_group': raw_group_path_str,
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

    def _find_best_raw_scale(self, target_label_scale, raw_attrs):
        """
        Finds the best raw scale level to use based on the target label scale.

        It prioritizes raw scales that are higher or equal resolution (smaller or equal scale value)
        than the target, picking the one with the closest resolution to minimize downsampling.
        """
        try:
            multiscales = raw_attrs['multiscales'][0]
            datasets = multiscales['datasets']
        except (KeyError, IndexError):
            return self.raw_scale, None, None

        available_scales = []
        for d in datasets:
            try:
                scale = next(t['scale'] for t in d['coordinateTransformations'] if t['type'] == 'scale')
                translation = next(t['translation'] for t in d['coordinateTransformations'] if t['type'] == 'translation')
                available_scales.append({'path': d['path'], 'scale': scale, 'translation': translation})
            except (KeyError, StopIteration):
                continue
        
        if not available_scales:
            return self.raw_scale, None, None

        # Find candidate scales where raw_resolution >= label_resolution (raw_scale <= label_scale)
        candidates = [s for s in available_scales if all(rs <= ls for rs, ls in zip(s['scale'], target_label_scale))]

        if candidates:
            # From the candidates, find the one closest to the target scale (minimizing the difference)
            best_candidate = min(candidates, key=lambda s: sum(ls - rs for ls, rs in zip(target_label_scale, s['scale'])))
            return best_candidate['path'], best_candidate['scale'], best_candidate['translation']
        else:
            # No suitable candidate for downsampling, so we'll have to upsample.
            # Pick the highest resolution available (smallest scale values).
            highest_res_scale = min(available_scales, key=lambda s: sum(s['scale']))
            return highest_res_scale['path'], highest_res_scale['scale'], highest_res_scale['translation']

    def __getitem__(self, idx):
        """
        Fetches a single raw crop and its corresponding segmentation mask, both at a fixed output size.
        """
        sample_info = self.samples[idx]
        zarr_path = sample_info['zarr_path']
        
        zarr_root = zarr.open(zarr_path, mode='r')
        
        label_array = zarr_root[sample_info['label_path']]

        # --- Parse Label Metadata ---
        label_attrs_group_path = os.path.dirname(sample_info['label_path'])
        label_attrs = zarr_root[label_attrs_group_path].attrs.asdict()
        label_scale_name = os.path.basename(sample_info['label_path'])
        label_scale, label_translation = self._parse_ome_ngff_metadata(label_attrs, label_scale_name)
        if label_scale is None:
             raise ValueError(f"Could not parse required OME-NGFF metadata from {label_attrs_group_path}")

        print(f"Zarr path: {sample_info['zarr_path']}")
        print(f"Label path: {sample_info['label_path']}, Label scale: {label_scale}, Label translation: {label_translation}")
        
        original_shape = label_array.shape
        target_shape = self.output_size

        # --- Adjust the label mask to the target output size ---
        # Case 1: The original label mask is larger than the target size, so we take a random crop.
        if all(os >= ts for os, ts in zip(original_shape, target_shape)):
            print(f"Cropping...")
            start_z = np.random.randint(0, original_shape[0] - target_shape[0] + 1)
            start_y = np.random.randint(0, original_shape[1] - target_shape[1] + 1)
            start_x = np.random.randint(0, original_shape[2] - target_shape[2] + 1)
            start_voxels_label = (start_z, start_y, start_x)

            slicing = tuple(slice(start, start + size) for start, size in zip(start_voxels_label, target_shape))
            final_label_mask = label_array[slicing]
            
            offset_physical = [start * scale for start, scale in zip(start_voxels_label, label_scale)]
            adjusted_label_translation = [orig + off for orig, off in zip(label_translation, offset_physical)]
            adjusted_label_scale = label_scale
        
        # Case 2: The label mask is smaller (or mixed), so we must resample it.
        else:
            print(f"Resampling...")
            label_data = label_array[:]
            zoom_factor = [t / s for t, s in zip(target_shape, original_shape)]
            # Use order=0 for nearest-neighbor interpolation to preserve integer labels
            resampled_label_mask = scipy.ndimage.zoom(label_data, zoom_factor, order=0, prefilter=False)
            
            final_label_mask = np.zeros(target_shape, dtype=resampled_label_mask.dtype)
            slicing_for_copy = tuple(slice(0, min(fs, cs)) for fs, cs in zip(target_shape, resampled_label_mask.shape))
            final_label_mask[slicing_for_copy] = resampled_label_mask[slicing_for_copy]

            adjusted_label_translation = label_translation
            original_physical_size = [sh * sc for sh, sc in zip(original_shape, label_scale)]
            adjusted_label_scale = [ps / ts for ps, ts in zip(original_physical_size, target_shape)]
            
        # --- Dynamically find the best raw scale and get its metadata ---
        raw_group_path = sample_info['raw_path_group']
        raw_attrs = zarr_root[raw_group_path].attrs.asdict()
        best_raw_scale_path, raw_scale, raw_translation = self._find_best_raw_scale(adjusted_label_scale, raw_attrs)
        print(f"Raw scale to retrieve: {raw_scale}")

        if raw_scale is None: # Fallback if metadata parsing failed in helper
             _, raw_scale, raw_translation = self._parse_ome_ngff_metadata(raw_attrs, best_raw_scale_path)
             if raw_scale is None:
                 print(f"Warning: Could not parse metadata for raw volume. Assuming scale=[1,1,1] and translation=[0,0,0].")
                 raw_scale, raw_translation = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]

        # --- Now fetch the corresponding raw data using the optimal raw scale ---
        best_raw_array_path = os.path.join(raw_group_path, best_raw_scale_path)
        raw_array = zarr_root[best_raw_array_path]

        scale_ratio = [ls / rs for ls, rs in zip(adjusted_label_scale, raw_scale)]
        relative_start_physical = [lt - rt for lt, rt in zip(adjusted_label_translation, raw_translation)]
        start_voxels_raw = [int(round(p / s)) for p, s in zip(relative_start_physical, raw_scale)]

        is_downsampling_or_equal = all(r >= 0.999 for r in scale_ratio)

        if is_downsampling_or_equal:
            print(f"Downsamplign...")
            step = [int(round(r)) for r in scale_ratio]
            step = [max(1, s) for s in step]
            end_voxels_raw = [st + (dim * sp) for st, dim, sp in zip(start_voxels_raw, target_shape, step)]
            slicing = tuple(slice(st, en, sp) for st, en, sp in zip(start_voxels_raw, end_voxels_raw, step))
            raw_crop_from_zarr = raw_array[slicing]
        else:
            print(f"Upsampling...")
            label_physical_size = [sh * sc for sh, sc in zip(target_shape, adjusted_label_scale)]
            relative_end_physical = [s + size for s, size in zip(relative_start_physical, label_physical_size)]
            end_voxels_raw = [int(round(p / s)) for p, s in zip(relative_end_physical, raw_scale)]
            slicing = tuple(slice(start, end) for start, end in zip(start_voxels_raw, end_voxels_raw))
            raw_crop = raw_array[slicing]

            if any(s == 0 for s in raw_crop.shape):
                raw_crop_from_zarr = np.zeros(target_shape, dtype=raw_array.dtype)
            else:
                zoom_factor = [t / s for t, s in zip(target_shape, raw_crop.shape)]
                raw_crop_from_zarr = scipy.ndimage.zoom(raw_crop, zoom_factor, order=1, prefilter=False)

        final_raw_crop = np.zeros(target_shape, dtype=raw_array.dtype)
        slicing_for_copy = tuple(slice(0, min(fs, cs)) for fs, cs in zip(target_shape, raw_crop_from_zarr.shape))
        final_raw_crop[slicing_for_copy] = raw_crop_from_zarr[slicing_for_copy]

        return final_raw_crop, final_label_mask


if __name__ == '__main__':
    
    DATA_DIR = '/lustre/gale/stf218/scratch/emin/cellmap-segmentation-challenge_old/data'

    print("\nInitializing ZarrSegmentationDataset...")
    # Initialize the dataset with the root directory containing the datasets.
    dataset = ZarrSegmentationDataset(root_dir=DATA_DIR)

    print(f"Found {len(dataset)} samples.")
    for i in range(len(dataset)):
        # Get the first sample
        print(f"Sample {i}")
        raw_image_crop, label_mask_crop = dataset[i]
        print(f"Raw image crop shape: {raw_image_crop.shape}")
        print(f"Label mask crop shape: {label_mask_crop.shape}")
        print(f"Unique labels: {np.unique(label_mask_crop)}")

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