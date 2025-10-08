import os
import zarr
import numpy as np
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
            raw_scale (str, optional): Resolution for raw data. Defaults to 's0' (highest resolution).
            labels_scale (str, optional): Resolution for labels. Defaults to 's0' (highest resolution).
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
        zarr_root = zarr.open(sample_info['zarr_path'], mode='r')
        label_array = zarr_root[sample_info['label_path']]

        # Parse label metadata
        label_attrs_group_path = os.path.dirname(sample_info['label_path'])
        label_attrs = zarr_root[label_attrs_group_path].attrs.asdict()
        label_scale_name = os.path.basename(sample_info['label_path'])
        label_scale, label_translation = self._parse_ome_ngff_metadata(label_attrs, label_scale_name)
        if label_scale is None:
             raise ValueError(f"Could not parse required OME-NGFF metadata from {label_attrs_group_path}")
        
        print(f"Zarr path: {sample_info['zarr_path']}")
        print(f"Label path: {sample_info['label_path']}, Label scale: {label_scale}, Label translation: {label_translation}")
    
        # Dynamically find the best raw scale based on the ORIGINAL label scale
        raw_group_path = sample_info['raw_path_group']
        raw_attrs = zarr_root[raw_group_path].attrs.asdict()
        best_raw_scale_path, raw_scale, raw_translation = self._find_best_raw_scale(label_scale, raw_attrs)
        print(f"Raw scale to retrieve: {raw_scale}")

        if raw_scale is None: # Fallback if metadata parsing failed in helper
             _, raw_scale, raw_translation = self._parse_ome_ngff_metadata(raw_attrs, best_raw_scale_path)
             if raw_scale is None:
                 print(f"Warning: Could not parse metadata for raw volume. Assuming scale=[1,1,1] and translation=[0,0,0].")
                 raw_scale, raw_translation = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
        
        original_shape = label_array.shape
        target_shape = self.output_size

        print(f"Original shape: {original_shape}")
        print(f"Target shape: {target_shape}")

        # ====== Adjust the label mask to the target output size ======
        # Case 1: The original label mask is larger than the target size, so we take a random crop.
        if all(os >= ts for os, ts in zip(original_shape, target_shape)):
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
            label_data = label_array[:]
            zoom_factor = [t / s for t, s in zip(target_shape, original_shape)]
            print(f"Zoom factor: {zoom_factor}")

            # Use order=0 for nearest-neighbor interpolation to preserve integer labels
            resampled_label_mask = scipy.ndimage.zoom(label_data, zoom_factor, order=0, prefilter=False)
            
            final_label_mask = np.zeros(target_shape, dtype=resampled_label_mask.dtype)
            slicing_for_copy = tuple(slice(0, min(fs, cs)) for fs, cs in zip(target_shape, resampled_label_mask.shape))
            final_label_mask[slicing_for_copy] = resampled_label_mask[slicing_for_copy]

            adjusted_label_translation = label_translation
            original_physical_size = [sh * sc for sh, sc in zip(original_shape, label_scale)]
            adjusted_label_scale = [ps / ts for ps, ts in zip(original_physical_size, target_shape)]
            
        # Now fetch the corresponding raw data using the optimal raw scale
        best_raw_array_path = os.path.join(raw_group_path, best_raw_scale_path)
        raw_array = zarr_root[best_raw_array_path]

        scale_ratio = [ls / rs for ls, rs in zip(adjusted_label_scale, raw_scale)]
        relative_start_physical = [lt - rt for lt, rt in zip(adjusted_label_translation, raw_translation)]
        start_voxels_raw = [int(round(p / s)) for p, s in zip(relative_start_physical, raw_scale)]

        is_downsampling_or_equal = all(r >= 0.999 for r in scale_ratio)

        if is_downsampling_or_equal:
            step = [int(round(r)) for r in scale_ratio]
            step = [max(1, s) for s in step]
            end_voxels_raw = [st + (dim * sp) for st, dim, sp in zip(start_voxels_raw, target_shape, step)]
            slicing = tuple(slice(st, en, sp) for st, en, sp in zip(start_voxels_raw, end_voxels_raw, step))
            raw_crop_from_zarr = raw_array[slicing]
        else:
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


class ZarrSegmentationDataset2D(ZarrSegmentationDataset):
    """
    A dataloader that provides 2D slices by directly slicing the Zarr store.

    This class treats the entire collection of 2D slices across all 3D volumes
    as one large dataset, enabling efficient, direct loading of 2D data without
    loading intermediate 3D chunks.
    """
    def __init__(self, root_dir, raw_scale='s0', labels_scale='s0', output_size=(512, 512)):
        """
        Initializes the 2D dataloader.

        Args:
            root_dir (str): The path to the root directory containing the Zarr datasets.
            raw_scale (str, optional): The default highest-resolution scale for raw data.
            labels_scale (str, optional): The scale level to use for the labels.
            output_size (tuple, optional): The desired (H, W) output size for the 2D slices.
        """
        super().__init__(root_dir, raw_scale, labels_scale)
        self.output_size = output_size
        self._build_slice_map()

    def _build_slice_map(self):
        """
        Scans all 3D volumes and builds a map of all possible 2D slices.
        This allows __len__ and __getitem__ to work on a virtual dataset of 2D slices.
        """
        print("Building 2D slice map... (This may take a moment for large datasets)")
        self.slice_map = []
        self.cumulative_slices = [0]
        total_slices = 0

        for sample_info in self.samples:
            zarr_root = zarr.open(sample_info['zarr_path'], mode='r')
            label_array = zarr_root[sample_info['label_path']]
            shape = label_array.shape
            
            # Add slices for Z, Y, and X axes
            for axis, num_slices in enumerate(shape):
                self.slice_map.append({'sample_info': sample_info, 'axis': axis})
                total_slices += num_slices
                self.cumulative_slices.append(total_slices)
        
        self.total_slices = total_slices
        print(f"Slice map built. Total 2D slices found: {self.total_slices}")

    def __len__(self):
        """Returns the total number of 2D slices available across all volumes."""
        return self.total_slices

    def __getitem__(self, idx):
        """
        Fetches a specific 2D slice by its global index, loading it directly from the Zarr store.
        """
        if not 0 <= idx < self.total_slices:
            raise IndexError("Index out of range")

        # Find which volume and axis this global index corresponds to
        map_idx = np.searchsorted(self.cumulative_slices, idx, side='right') - 1
        slice_info = self.slice_map[map_idx]
        local_slice_idx = idx - self.cumulative_slices[map_idx]

        sample_info = slice_info['sample_info']
        axis = slice_info['axis']

        # Get label slice and metadata
        zarr_root = zarr.open(sample_info['zarr_path'], mode='r')
        label_array_3d = zarr_root[sample_info['label_path']]
        
        slicing_3d = [slice(None)] * 3
        slicing_3d[axis] = local_slice_idx
        label_slice_2d = label_array_3d[tuple(slicing_3d)]

        label_attrs_group_path = os.path.dirname(sample_info['label_path'])
        label_attrs = zarr_root[label_attrs_group_path].attrs.asdict()
        label_scale_name = os.path.basename(sample_info['label_path'])
        label_scale_3d, label_translation_3d = self._parse_ome_ngff_metadata(label_attrs, label_scale_name)

        # Adjust label slice to output_size (crop or resample)
        original_shape_2d = label_slice_2d.shape
        target_shape_2d = self.output_size
        print(f"Original shape: {original_shape_2d}; target shape: {target_shape_2d}")

        if all(os >= ts for os, ts in zip(original_shape_2d, target_shape_2d)):
            print(f"Cropping...")
            start_h = np.random.randint(0, original_shape_2d[0] - target_shape_2d[0] + 1)
            start_w = np.random.randint(0, original_shape_2d[1] - target_shape_2d[1] + 1)
            final_label_slice = label_slice_2d[start_h:start_h+target_shape_2d[0], start_w:start_w+target_shape_2d[1]]
            
            # Adjust metadata for the crop
            axes_2d = [i for i in range(3) if i != axis]
            offset_physical = [start_h * label_scale_3d[axes_2d[0]], start_w * label_scale_3d[axes_2d[1]]]
            adjusted_label_translation_2d = [label_translation_3d[axes_2d[0]] + offset_physical[0], label_translation_3d[axes_2d[1]] + offset_physical[1]]
            adjusted_label_scale_2d = [label_scale_3d[axes_2d[0]], label_scale_3d[axes_2d[1]]]
        else:
            print(f"Resampling...")
            zoom_factor = [t / s for t, s in zip(target_shape_2d, original_shape_2d)]
            final_label_slice = scipy.ndimage.zoom(label_slice_2d, zoom_factor, order=0, prefilter=False)
            
            # Adjust metadata for the resampling
            axes_2d = [i for i in range(3) if i != axis]
            adjusted_label_translation_2d = [label_translation_3d[axes_2d[0]], label_translation_3d[axes_2d[1]]]
            original_physical_size_2d = [sh * sc for sh, sc in zip(original_shape_2d, [label_scale_3d[d] for d in axes_2d])]
            adjusted_label_scale_2d = [ps / ts for ps, ts in zip(original_physical_size_2d, target_shape_2d)]

        # Find best raw scale
        raw_group_path = sample_info['raw_path_group']
        raw_attrs = zarr_root[raw_group_path].attrs.asdict()
        
        # We need a 3D target scale to compare with the raw scales
        temp_target_label_scale_3d = [0,0,0]
        axes_2d = [i for i in range(3) if i != axis]
        temp_target_label_scale_3d[axes_2d[0]] = adjusted_label_scale_2d[0]
        temp_target_label_scale_3d[axes_2d[1]] = adjusted_label_scale_2d[1]
        temp_target_label_scale_3d[axis] = label_scale_3d[axis] # thickness of the slice

        best_raw_scale_path, raw_scale_3d, raw_translation_3d = self._find_best_raw_scale(temp_target_label_scale_3d, raw_attrs)
        
        # Calculate raw slice coordinates
        raw_array_3d = zarr_root[os.path.join(raw_group_path, best_raw_scale_path)]
        
        # Construct 3D physical coordinates for the 2D plane
        phys_start_3d = [0,0,0]
        phys_start_3d[axes_2d[0]] = adjusted_label_translation_2d[0]
        phys_start_3d[axes_2d[1]] = adjusted_label_translation_2d[1]
        phys_start_3d[axis] = label_translation_3d[axis] + local_slice_idx * label_scale_3d[axis]
        
        # Convert to raw voxel coordinates
        relative_phys_start_3d = [ps - rt for ps, rt in zip(phys_start_3d, raw_translation_3d)]
        start_voxels_raw_3d = [int(round(p / s)) for p, s in zip(relative_phys_start_3d, raw_scale_3d)]

        # Determine the size of the slice in raw voxels
        size_in_phys_2d = [sh * sc for sh, sc in zip(target_shape_2d, adjusted_label_scale_2d)]
        size_in_raw_voxels_2d = [int(round(p / s)) for p, s in zip(size_in_phys_2d, [raw_scale_3d[d] for d in axes_2d])]
        
        # Construct the final 3D slice for the raw array
        raw_slicing = [0,0,0]
        raw_slicing[axes_2d[0]] = slice(start_voxels_raw_3d[axes_2d[0]], start_voxels_raw_3d[axes_2d[0]] + size_in_raw_voxels_2d[0])
        raw_slicing[axes_2d[1]] = slice(start_voxels_raw_3d[axes_2d[1]], start_voxels_raw_3d[axes_2d[1]] + size_in_raw_voxels_2d[1])
        raw_slicing[axis] = start_voxels_raw_3d[axis]

        raw_slice_2d = raw_array_3d[tuple(raw_slicing)]

        # Final resampling of raw slice
        if raw_slice_2d.shape != target_shape_2d:
             if any(s == 0 for s in raw_slice_2d.shape):
                 final_raw_slice = np.zeros(target_shape_2d, dtype=raw_array_3d.dtype)
             else:
                zoom_factor = [t / s for t, s in zip(target_shape_2d, raw_slice_2d.shape)]
                final_raw_slice = scipy.ndimage.zoom(raw_slice_2d, zoom_factor, order=1, prefilter=False)
        else:
            final_raw_slice = raw_slice_2d

        return final_raw_slice, final_label_slice


if __name__ == '__main__':
    
    DATA_DIR = '/lustre/gale/stf218/scratch/emin/cellmap-segmentation-challenge_old/data'

    # ====== Test 3D dataset class ======
    print("\nInitializing ZarrSegmentationDataset...")
    # Initialize the dataset with the root directory containing the datasets.
    dataset = ZarrSegmentationDataset(root_dir=DATA_DIR)

    # print(f"Found {len(dataset)} samples.")
    # for i in range(len(dataset)):
    #     # Get the first sample
    #     print(f"Sample {i}")
    #     raw_image_crop, label_mask_crop = dataset[i]
    #     print(f"Raw image crop shape: {raw_image_crop.shape}")
    #     print(f"Label mask crop shape: {label_mask_crop.shape}")
    #     print(f"Unique labels: {np.unique(label_mask_crop)}")

    print("\nFetching the first sample...")
    # Get the first sample
    crop_idx = 0
    raw_image_crop, label_mask_crop = dataset[crop_idx]

    print(f"Raw image crop shape: {raw_image_crop.shape}")
    print(f"Label mask crop shape: {label_mask_crop.shape}")
    print(f"Unique labels: {np.unique(label_mask_crop)}")

    # Verify the shapes match
    assert raw_image_crop.shape == label_mask_crop.shape
    print("\nSuccessfully loaded a sample and shapes match!")

    # ====== visualization check ======
    import matplotlib.pyplot as plt

    num_slices_to_show = 36
    fig, axes = plt.subplots(int(np.sqrt(num_slices_to_show)), int(np.sqrt(num_slices_to_show)), figsize=(20, 20))
    fig.suptitle(f"Crop index: {crop_idx}", fontsize=16)
    slice_indices = np.linspace(0, raw_image_crop.shape[0] - 1, num_slices_to_show, dtype=int)

    # flatten array for easy iteration
    axes = axes.flatten()

    vmin, vmax = 0, np.max(label_mask_crop)

    for i, slice_idx in enumerate(slice_indices):
        ax = axes[i]
        # Display the raw image slice
        ax.imshow(raw_image_crop[slice_idx, :, :], cmap='gray')
        
        # Overlay the label mask, use masked array to make the background (label 0) transparent
        masked_labels = np.ma.masked_where(label_mask_crop[slice_idx, :, :] == 0, label_mask_crop[slice_idx, :, :])
        ax.imshow(masked_labels, cmap='gist_ncar', alpha=0.1, interpolation='none', vmin=vmin, vmax=vmax)
        
        ax.set_title(f'Slice Z={slice_idx}')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"raw_labeled_crop_{crop_idx}.jpeg", bbox_inches='tight')
    print("Figure successfully saved.")

    # # ====== Test 2D dataset class ======
    # print("\nInitializing ZarrSegmentationDataset2D...")
    # fixed_output_size_2d = (512, 512)
    # dataset_2d = ZarrSegmentationDataset2D(root_dir=DATA_DIR, output_size=fixed_output_size_2d)

    # print(f"Total 2D slices available: {len(dataset_2d)}")
    # # Fetch a slice to test
    # random_slice_idx = 0  #np.random.randint(0, len(dataset_2d))
    # print(f"Fetching random slice index {random_slice_idx}...")
    # raw_slice, label_slice = dataset_2d[random_slice_idx]
    # print(f"Final Raw 2D Shape: {raw_slice.shape}")
    # print(f"Final Label 2D Shape: {label_slice.shape}")

    # assert raw_slice.shape == fixed_output_size_2d
    # assert label_slice.shape == fixed_output_size_2d
    # print("2D shapes match the target output size correctly.")

    # # # ====== visualization check ======
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # fig.suptitle('Visual Correspondence Check (2D Direct Slice)', fontsize=16)
    
    # # Display the raw image slice
    # ax.imshow(raw_slice, cmap='gray')
    
    # # Overlay the label mask
    # masked_labels = np.ma.masked_where(label_slice == 0, label_slice)
    # ax.imshow(masked_labels, cmap='gist_ncar', alpha=0.1, interpolation='none')
    
    # ax.set_title('Random 2D Slice with Overlay')
    # ax.axis('off')

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig(f"raw_labeled_crop_2D_{random_slice_idx}.jpeg", bbox_inches='tight')
    # print("Figure successfully saved.")
