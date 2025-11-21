import os
import zarr
import numpy as np
from glob import glob
import scipy.ndimage

class ZarrSegmentationDataset:
    def __init__(self, root_dir, raw_scale='s0', labels_scale='s0', output_size=(512, 512, 512)):
        self.root_dir = root_dir
        self.raw_scale = raw_scale
        self.labels_scale = labels_scale
        self.output_size = output_size
        
        # GLOBAL MAPPING: Name (str) -> Unified Master ID (int)
        # We start with 0 always being "unlabeled" or "background" if desired, 
        # but here we will build it dynamically based on strings found.
        self.name_to_master_id = {}
        self.master_id_to_name = {}
        
        self.samples = self._find_samples()

        print(f"\nDataset Initialized with {len(self.samples)} samples.")
        print(f"Total Unique Classes Found Globally: {len(self.name_to_master_id)}")
        print(f"Global Class List: {self.name_to_master_id}")

    def _find_samples(self):
        samples = []
        zarr_paths = glob(os.path.join(self.root_dir, '*/*.zarr'))

        for zarr_path in zarr_paths:
            try:
                zarr_root = zarr.open(zarr_path, mode='r')
            except Exception as e:
                print(f"Skipping {zarr_path}: {e}")
                continue

            for recon_name in zarr_root.keys():
                if not recon_name.startswith('recon-'): continue

                raw_group_path_str = os.path.join(recon_name, 'em', 'fibsem-uint8')
                labels_base_path_str = os.path.join(recon_name, 'labels', 'groundtruth')

                if raw_group_path_str not in zarr_root or labels_base_path_str not in zarr_root:
                    continue
                
                label_base_group = zarr_root[labels_base_path_str]

                for crop_name in label_base_group.keys():
                    if not crop_name.startswith('crop'): continue
                    
                    # 1. Get the specific metadata for THIS crop
                    crop_node = label_base_group[crop_name]
                    local_class_names = self._get_crop_class_names(crop_node)
                    
                    # If no metadata, we can't safely use this crop in a mixed dataset
                    if not local_class_names:
                        continue

                    # 2. Create a "Local ID" -> "Master ID" lookup table (LUT)
                    # This maps the inconsistent file integers to our consistent global integers
                    remap_lut = self._create_remap_lut(local_class_names)

                    label_path_str = os.path.join(labels_base_path_str, crop_name, 'all', self.labels_scale)
                    
                    if label_path_str in zarr_root:
                        samples.append({
                            'zarr_path': zarr_path,
                            'raw_path_group': raw_group_path_str,
                            'label_path': label_path_str,
                            'remap_lut': remap_lut  # Store the translation table
                        })

        return samples

    def _get_crop_class_names(self, crop_node):
        """Extracts the list of class names for a specific crop."""
        attrs = crop_node.attrs.asdict()
        try:
            return attrs['cellmap']['annotation']['class_names']
        except KeyError:
            return None

    def _create_remap_lut(self, local_names):
        """
        Builds a numpy array where array[local_id] = master_id.
        Also updates the global self.name_to_master_id registry.
        """
        # Determine the maximum possible integer in this local file (indices of the list)
        max_local_id = len(local_names) - 1
        
        # Create a lookup table. Default to 0 (or a specific 'unknown' ID)
        lut = np.zeros(max_local_id + 1, dtype=np.int64)

        for local_id, name in enumerate(local_names):
            # Register name globally if new
            if name not in self.name_to_master_id:
                new_master_id = len(self.name_to_master_id)
                self.name_to_master_id[name] = new_master_id
                self.master_id_to_name[new_master_id] = name
            
            # Map local -> global
            master_id = self.name_to_master_id[name]
            lut[local_id] = master_id
            
        return lut

    # ... [Include _parse_ome_ngff_metadata and _find_best_raw_scale here] ...
    # (These helper functions remain exactly the same as your original code)
    def _parse_ome_ngff_metadata(self, attrs, scale_level_name):
        try:
            multiscales = attrs['multiscales'][0]
            datasets = multiscales['datasets']
            scale_metadata = next((d for d in datasets if d['path'] == scale_level_name), None)
            if scale_metadata:
                transformations = scale_metadata['coordinateTransformations']
                scale_transform = next((t for t in transformations if t['type'] == 'scale'), None)
                translation_transform = next((t for t in transformations if t['type'] == 'translation'), None)
                return (scale_transform['scale'] if scale_transform else [1.0, 1.0, 1.0]), \
                       (translation_transform['translation'] if translation_transform else [0.0, 0.0, 0.0])
        except (KeyError, IndexError, StopIteration): pass
        return None, None

    def _find_best_raw_scale(self, target_label_scale, raw_attrs):
        try:
            multiscales = raw_attrs['multiscales'][0]
            datasets = multiscales['datasets']
        except (KeyError, IndexError): return self.raw_scale, None, None
        available_scales = []
        for d in datasets:
            try:
                scale = next(t['scale'] for t in d['coordinateTransformations'] if t['type'] == 'scale')
                translation = next(t['translation'] for t in d['coordinateTransformations'] if t['type'] == 'translation')
                available_scales.append({'path': d['path'], 'scale': scale, 'translation': translation})
            except (KeyError, StopIteration): continue
        if not available_scales: return self.raw_scale, None, None
        candidates = [s for s in available_scales if all(rs <= ls for rs, ls in zip(s['scale'], target_label_scale))]
        if candidates:
            best = min(candidates, key=lambda s: sum(ls - rs for ls, rs in zip(target_label_scale, s['scale'])))
            return best['path'], best['scale'], best['translation']
        best = min(available_scales, key=lambda s: sum(s['scale']))
        return best['path'], best['scale'], best['translation']

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]        
        zarr_root = zarr.open(sample_info['zarr_path'], mode='r')
        
        # 1. Load Raw Label Data (Local IDs)
        label_array = zarr_root[sample_info['label_path']]
        
        # --- Standard Metadata Parsing (Same as before) ---
        label_attrs_group_path = os.path.dirname(sample_info['label_path'])
        label_attrs = zarr_root[label_attrs_group_path].attrs.asdict()
        label_scale_name = os.path.basename(sample_info['label_path'])
        label_scale, label_translation = self._parse_ome_ngff_metadata(label_attrs, label_scale_name)
        if label_scale is None: label_scale, label_translation = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]

        raw_group_path = sample_info['raw_path_group']
        raw_attrs = zarr_root[raw_group_path].attrs.asdict()
        best_raw_scale_path, raw_scale, raw_translation = self._find_best_raw_scale(label_scale, raw_attrs)
        if raw_scale is None: 
             _, raw_scale, raw_translation = self._parse_ome_ngff_metadata(raw_attrs, best_raw_scale_path)
             if raw_scale is None: raw_scale, raw_translation = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]

        # --- Crop Logic (Simplified for brevity, insert your full crop logic here) ---
        original_shape = label_array.shape
        target_shape = self.output_size
        
        # Note: You should perform the random crop calculation here
        # For demonstration, we take a center crop
        slicing = tuple(slice(0, min(o, t)) for o, t in zip(original_shape, target_shape))
        
        # Load data into memory
        local_label_mask = label_array[slicing]
        local_label_mask = np.array(local_label_mask) # Ensure numpy array
        
        # ============================================================
        # CRITICAL STEP: REMAP TO GLOBAL IDs
        # ============================================================
        # We use NumPy advanced indexing to swap values instantly.
        # values in local_label_mask are used as indices into the LUT.
        # e.g. if pixel is 58, it grabs index 58 from LUT, which might be 102 (Global Actin)
        
        # Handle potential out-of-bounds (if a pixel value exists > len(lut))
        # This happens if metadata list is shorter than actual pixel values used
        lut = sample_info['remap_lut']
        max_lut_idx = len(lut) - 1
        
        # Safety clip to prevent crashing if a pixel value exceeds the metadata list
        # (Maps unknown high values to the last entry, or 0 if you prefer)
        safe_indices = np.clip(local_label_mask, 0, max_lut_idx)
        
        global_label_mask = lut[safe_indices] 
        
        # ============================================================

        # Load Raw Data
        raw_array = zarr_root[os.path.join(raw_group_path, best_raw_scale_path)]
        final_raw_crop = raw_array[slicing] # (Apply your coordinate math here for alignment)

        # Pad if necessary (if crop was smaller than target)
        if global_label_mask.shape != target_shape:
            padded_label = np.zeros(target_shape, dtype=global_label_mask.dtype)
            padded_raw = np.zeros(target_shape, dtype=final_raw_crop.dtype)
            
            sl = tuple(slice(0, s) for s in global_label_mask.shape)
            padded_label[sl] = global_label_mask
            padded_raw[sl] = final_raw_crop
            
            return padded_raw, padded_label

        return final_raw_crop, global_label_mask


# class ZarrSegmentationDataset:
#     """
#     A simple dataloader for 3D EM segmentation datasets stored in Zarr format.

#     This dataset class scans a root directory to find pairs of raw EM volumes
#     and their corresponding labeled segmentation crops. It returns fixed-size
#     crops suitable for training deep learning models.
#     """
#     def __init__(self, root_dir, raw_scale='s0', labels_scale='s0', output_size=(512, 512, 512)):
#         """
#         Initializes the dataset by scanning for valid data samples.

#         Args:
#             root_dir (str): The path to the root directory containing the Zarr datasets.
#             raw_scale (str, optional): Resolution for raw data. Defaults to 's0' (highest resolution).
#             labels_scale (str, optional): Resolution for labels. Defaults to 's0' (highest resolution).
#             output_size (tuple, optional): The desired (Z, Y, X) output size of the raw and label crops. Defaults to (512, 512, 512).
#         """
#         self.root_dir = root_dir
#         self.raw_scale = raw_scale
#         self.labels_scale = labels_scale
#         self.output_size = output_size
#         self.id_to_name = {}  # Dictionary mapping Label ID (int) -> Label Name (str) (this will be populated upon calling self._find_samples() below)
#         self.samples = self._find_samples()

#         if not self.samples:
#             print(f"Warning: No valid samples found in {self.root_dir}. Please check the directory structure and file paths.")

#     def _find_samples(self):
#         samples = []
#         zarr_paths = glob(os.path.join(self.root_dir, '*/*.zarr'))

#         for zarr_path in zarr_paths:
#             try:
#                 zarr_root = zarr.open(zarr_path, mode='r')
#             except Exception as e:
#                 print(f"Could not open {zarr_path}, skipping. Error: {e}")
#                 continue

#             for recon_name in zarr_root.keys():
#                 if not recon_name.startswith('recon-'):
#                     continue

#                 raw_group_path_str = os.path.join(recon_name, 'em', 'fibsem-uint8')
#                 labels_base_path_str = os.path.join(recon_name, 'labels', 'groundtruth')

#                 if raw_group_path_str not in zarr_root or labels_base_path_str not in zarr_root:
#                     continue
                
#                 label_base_group = zarr_root[labels_base_path_str]

#                 # Iterate through CROPS (crop1, crop234, etc.)
#                 for crop_name in label_base_group.keys():
#                     if not crop_name.startswith('crop'):
#                         continue
                    
#                     # --- NEW: Extract metadata from the CROP level ---
#                     # We pass the specific crop group (e.g., groundtruth/crop234)
#                     crop_node = label_base_group[crop_name]
#                     self._extract_label_names(crop_node, source_name=f"{os.path.basename(zarr_path)}/{crop_name}")

#                     label_path_str = os.path.join(labels_base_path_str, crop_name, 'all', self.labels_scale)
                    
#                     if label_path_str in zarr_root:
#                         samples.append({
#                             'zarr_path': zarr_path,
#                             'raw_path_group': raw_group_path_str,
#                             'label_path': label_path_str
#                         })

#         return samples

#     def _extract_label_names(self, group_node, source_name="Unknown"):
#         """
#         Parses CellMap/OpenOrganelle specific metadata attributes.
#         Updates self.id_to_name globally and checks for consistency across crops.
#         """
#         attrs = group_node.attrs.asdict()
        
#         # Access nested keys safely: cellmap -> annotation -> class_names
#         try:
#             # This matches the JSON structure you provided
#             class_names = attrs['cellmap']['annotation']['class_names']
#         except KeyError:
#             # If this crop doesn't have the metadata, just skip it
#             return

#         if isinstance(class_names, list):
#             for idx, name in enumerate(class_names):
#                 # Standard assumption: List index = Integer Label ID
#                 # "ecs" is at index 0, so pixel value 0 = "ecs"
                
#                 # Consistency Check:
#                 # If we have seen this ID before, ensure the name matches.
#                 if idx in self.id_to_name:
#                     existing_name = self.id_to_name[idx]
#                     if existing_name != name:
#                         # Only warn if they are strictly different
#                         print(f"WARNING: Label Conflict for ID {idx}!")
#                         print(f"  Existing: '{existing_name}'")
#                         print(f"  New ({source_name}): '{name}'")
#                         print(f"  Keeping '{existing_name}' for now.")
#                 else:
#                     self.id_to_name[idx] = name

#     def __len__(self):
#         """Returns the total number of samples (label crops) found."""
#         return len(self.samples)
    
#     def _parse_ome_ngff_metadata(self, attrs, scale_level_name):
#         """Helper function to parse scale and translation from OME-NGFF metadata."""
#         try:
#             multiscales = attrs['multiscales'][0]
#             datasets = multiscales['datasets']
#             scale_metadata = next((d for d in datasets if d['path'] == scale_level_name), None)
            
#             if scale_metadata:
#                 transformations = scale_metadata['coordinateTransformations']
#                 scale_transform = next((t for t in transformations if t['type'] == 'scale'), None)
#                 translation_transform = next((t for t in transformations if t['type'] == 'translation'), None)
                
#                 scale = scale_transform['scale'] if scale_transform else [1.0, 1.0, 1.0]
#                 translation = translation_transform['translation'] if translation_transform else [0.0, 0.0, 0.0]
                
#                 return scale, translation
#         except (KeyError, IndexError, StopIteration):
#             pass # We will handle the error outside this function
        
#         return None, None
    
#     def _find_best_raw_scale(self, target_label_scale, raw_attrs):
#         """
#         Finds the best raw scale level to use based on the target label scale.

#         It prioritizes raw scales that are higher or equal resolution (smaller or equal scale value)
#         than the target, picking the one with the closest resolution to minimize downsampling.
#         """
#         try:
#             multiscales = raw_attrs['multiscales'][0]
#             datasets = multiscales['datasets']
#         except (KeyError, IndexError):
#             return self.raw_scale, None, None

#         available_scales = []
#         for d in datasets:
#             try:
#                 scale = next(t['scale'] for t in d['coordinateTransformations'] if t['type'] == 'scale')
#                 translation = next(t['translation'] for t in d['coordinateTransformations'] if t['type'] == 'translation')
#                 available_scales.append({'path': d['path'], 'scale': scale, 'translation': translation})
#             except (KeyError, StopIteration):
#                 continue
        
#         if not available_scales:
#             return self.raw_scale, None, None

#         # Find candidate scales where raw_resolution >= label_resolution (raw_scale <= label_scale)
#         candidates = [s for s in available_scales if all(rs <= ls for rs, ls in zip(s['scale'], target_label_scale))]

#         if candidates:
#             # From the candidates, find the one closest to the target scale (minimizing the difference)
#             best_candidate = min(candidates, key=lambda s: sum(ls - rs for ls, rs in zip(target_label_scale, s['scale'])))
#             return best_candidate['path'], best_candidate['scale'], best_candidate['translation']
#         else:
#             # No suitable candidate for downsampling, so we'll have to upsample.
#             # Pick the highest resolution available (smallest scale values).
#             highest_res_scale = min(available_scales, key=lambda s: sum(s['scale']))
#             return highest_res_scale['path'], highest_res_scale['scale'], highest_res_scale['translation']

#     def __getitem__(self, idx):
#         """
#         Fetches a single raw crop and its corresponding segmentation mask, both at a fixed output size.
#         """
#         sample_info = self.samples[idx]        
#         zarr_root = zarr.open(sample_info['zarr_path'], mode='r')
#         label_array = zarr_root[sample_info['label_path']]

#         # Parse label metadata
#         label_attrs_group_path = os.path.dirname(sample_info['label_path'])
#         label_attrs = zarr_root[label_attrs_group_path].attrs.asdict()
#         label_scale_name = os.path.basename(sample_info['label_path'])
#         label_scale, label_translation = self._parse_ome_ngff_metadata(label_attrs, label_scale_name)
#         if label_scale is None:
#              raise ValueError(f"Could not parse required OME-NGFF metadata from {label_attrs_group_path}")
        
#         print(f"Zarr path: {sample_info['zarr_path']}")
#         print(f"Label path: {sample_info['label_path']}, Label scale: {label_scale}, Label translation: {label_translation}")
    
#         # Dynamically find the best raw scale based on the ORIGINAL label scale
#         raw_group_path = sample_info['raw_path_group']
#         raw_attrs = zarr_root[raw_group_path].attrs.asdict()
#         best_raw_scale_path, raw_scale, raw_translation = self._find_best_raw_scale(label_scale, raw_attrs)
#         print(f"Raw scale to retrieve: {raw_scale}")

#         if raw_scale is None: # Fallback if metadata parsing failed in helper
#              _, raw_scale, raw_translation = self._parse_ome_ngff_metadata(raw_attrs, best_raw_scale_path)
#              if raw_scale is None:
#                  print(f"Warning: Could not parse metadata for raw volume. Assuming scale=[1,1,1] and translation=[0,0,0].")
#                  raw_scale, raw_translation = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]
        
#         original_shape = label_array.shape
#         target_shape = self.output_size
#         label_array_np = label_array[:]

#         print(f"Label array min/max: {label_array_np.min()}/{label_array_np.max()}")
#         print(f"Label array shape (original): {original_shape}")
#         print(f"Label array shape (target): {target_shape}")

#         # ====== Adjust the label mask to the target output size ======
#         # Case 1: The original label mask is larger than the target size, so we take a random crop.
#         if all(os >= ts for os, ts in zip(original_shape, target_shape)):
#             start_z = np.random.randint(0, original_shape[0] - target_shape[0] + 1)
#             start_y = np.random.randint(0, original_shape[1] - target_shape[1] + 1)
#             start_x = np.random.randint(0, original_shape[2] - target_shape[2] + 1)
#             start_voxels_label = (start_z, start_y, start_x)

#             slicing = tuple(slice(start, start + size) for start, size in zip(start_voxels_label, target_shape))
#             final_label_mask = label_array[slicing]
            
#             offset_physical = [start * scale for start, scale in zip(start_voxels_label, label_scale)]
#             adjusted_label_translation = [orig + off for orig, off in zip(label_translation, offset_physical)]
#             adjusted_label_scale = label_scale
#         # Case 2: The label mask is smaller (or mixed), so we must resample it.
#         else:
#             label_data = label_array[:]
#             zoom_factor = [t / s for t, s in zip(target_shape, original_shape)]
#             print(f"Zoom factor: {zoom_factor}")

#             # Use order=0 for nearest-neighbor interpolation to preserve integer labels
#             resampled_label_mask = scipy.ndimage.zoom(label_data, zoom_factor, order=0, prefilter=False)
            
#             final_label_mask = np.zeros(target_shape, dtype=resampled_label_mask.dtype)
#             slicing_for_copy = tuple(slice(0, min(fs, cs)) for fs, cs in zip(target_shape, resampled_label_mask.shape))
#             final_label_mask[slicing_for_copy] = resampled_label_mask[slicing_for_copy]

#             adjusted_label_translation = label_translation
#             original_physical_size = [sh * sc for sh, sc in zip(original_shape, label_scale)]
#             adjusted_label_scale = [ps / ts for ps, ts in zip(original_physical_size, target_shape)]
            
#         # Now fetch the corresponding raw data using the optimal raw scale
#         best_raw_array_path = os.path.join(raw_group_path, best_raw_scale_path)
#         raw_array = zarr_root[best_raw_array_path]

#         scale_ratio = [ls / rs for ls, rs in zip(adjusted_label_scale, raw_scale)]
#         relative_start_physical = [lt - rt for lt, rt in zip(adjusted_label_translation, raw_translation)]
#         start_voxels_raw = [int(round(p / s)) for p, s in zip(relative_start_physical, raw_scale)]

#         is_downsampling_or_equal = all(r >= 0.999 for r in scale_ratio)

#         if is_downsampling_or_equal:
#             step = [int(round(r)) for r in scale_ratio]
#             step = [max(1, s) for s in step]
#             end_voxels_raw = [st + (dim * sp) for st, dim, sp in zip(start_voxels_raw, target_shape, step)]
#             slicing = tuple(slice(st, en, sp) for st, en, sp in zip(start_voxels_raw, end_voxels_raw, step))
#             raw_crop_from_zarr = raw_array[slicing]
#         else:
#             label_physical_size = [sh * sc for sh, sc in zip(target_shape, adjusted_label_scale)]
#             relative_end_physical = [s + size for s, size in zip(relative_start_physical, label_physical_size)]
#             end_voxels_raw = [int(round(p / s)) for p, s in zip(relative_end_physical, raw_scale)]
#             slicing = tuple(slice(start, end) for start, end in zip(start_voxels_raw, end_voxels_raw))
#             raw_crop = raw_array[slicing]

#             if any(s == 0 for s in raw_crop.shape):
#                 raw_crop_from_zarr = np.zeros(target_shape, dtype=raw_array.dtype)
#             else:
#                 zoom_factor = [t / s for t, s in zip(target_shape, raw_crop.shape)]
#                 raw_crop_from_zarr = scipy.ndimage.zoom(raw_crop, zoom_factor, order=1, prefilter=False)

#         final_raw_crop = np.zeros(target_shape, dtype=raw_array.dtype)
#         slicing_for_copy = tuple(slice(0, min(fs, cs)) for fs, cs in zip(target_shape, raw_crop_from_zarr.shape))
#         final_raw_crop[slicing_for_copy] = raw_crop_from_zarr[slicing_for_copy]

#         return final_raw_crop, final_label_mask


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
        print(f"Min/max raw: {final_raw_slice.min()}/{final_raw_slice.max()}")
        print(f"Min/max label: {final_label_slice.min()}/{final_label_slice.max()}")
        return final_raw_slice, final_label_slice


if __name__ == '__main__':
    
    DATA_DIR = '/lustre/gale/stf218/scratch/emin/cellmap-segmentation-challenge/data'

    # ====== Test 3D dataset class ======
    print("\nInitializing ZarrSegmentationDataset...")
    # Initialize the dataset with the root directory containing the datasets.
    dataset = ZarrSegmentationDataset(root_dir=DATA_DIR)

    # # Check metadata extraction immediately (Fast)
    # print("\nMetadata Extraction...")
    # print("Label Names Found in Attributes:", dataset.id_to_name)

    # print(f"Found {len(dataset)} samples.")
    # for i in range(len(dataset)):
    #     # Get the first sample
    #     print(f"Sample {i}")
    #     raw_image_crop, label_mask_crop = dataset[i]
    #     print(f"Raw image crop shape: {raw_image_crop.shape}")
    #     print(f"Label mask crop shape: {label_mask_crop.shape}")
    #     print(f"Unique labels: {np.unique(label_mask_crop)}")

    # print("\nFetching the first sample...")
    # # Get the first sample
    # crop_idx = 0
    # raw_image_crop, label_mask_crop = dataset[crop_idx]

    # print(f"Raw image crop shape: {raw_image_crop.shape}")
    # print(f"Label mask crop shape: {label_mask_crop.shape}")
    # print(f"Unique labels: {np.unique(label_mask_crop)}")

    # # Verify the shapes match
    # assert raw_image_crop.shape == label_mask_crop.shape
    # print("\nSuccessfully loaded a sample and shapes match!")

    # # Visualization check
    # import matplotlib.pyplot as plt

    # num_slices_to_show = 36
    # fig, axes = plt.subplots(int(np.sqrt(num_slices_to_show)), int(np.sqrt(num_slices_to_show)), figsize=(20, 20))
    # fig.suptitle(f"Crop index: {crop_idx}", fontsize=16)
    # slice_indices = np.linspace(0, raw_image_crop.shape[0] - 1, num_slices_to_show, dtype=int)

    # # flatten array for easy iteration
    # axes = axes.flatten()

    # vmin, vmax = 0, np.max(label_mask_crop)

    # for i, slice_idx in enumerate(slice_indices):
    #     ax = axes[i]
    #     # Display the raw image slice
    #     ax.imshow(raw_image_crop[slice_idx, :, :], cmap='gray')
        
    #     # Overlay the label mask, use masked array to make the background (label 0) transparent
    #     masked_labels = np.ma.masked_where(label_mask_crop[slice_idx, :, :] == 0, label_mask_crop[slice_idx, :, :])
    #     ax.imshow(masked_labels, cmap='gist_ncar', alpha=0.1, interpolation='none', vmin=vmin, vmax=vmax)
        
    #     ax.set_title(f'Slice Z={slice_idx}')
    #     ax.axis('off')

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig(f"raw_labeled_crop_{crop_idx}.jpeg", bbox_inches='tight')
    # print("Figure successfully saved.")

    # # # ====== Test 2D dataset class ======
    # print("\nInitializing ZarrSegmentationDataset2D...")
    # fixed_output_size_2d = (1024, 1024)
    # dataset_2d = ZarrSegmentationDataset2D(root_dir=DATA_DIR, output_size=fixed_output_size_2d)

    # # Visualization check
    # import matplotlib.pyplot as plt
    # num_slices_to_show = 100
    # n_cols = int(np.ceil(np.sqrt(num_slices_to_show)))
    # n_rows = int(np.ceil(num_slices_to_show / n_cols))

    # fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))
    
    # # Get regularly spaced indices from the entire 2D dataset
    # slice_indices = np.linspace(0, len(dataset_2d) - 1, num_slices_to_show, dtype=int)

    # for i, slice_idx in enumerate(slice_indices):
    #     ax = axes.flat[i]
    #     raw_slice, label_slice = dataset_2d[slice_idx]
        
    #     # Display the raw image slice
    #     ax.imshow(raw_slice, cmap='gray')
        
    #     # Overlay the label mask
    #     masked_labels = np.ma.masked_where(label_slice == 0, label_slice)
    #     ax.imshow(masked_labels, cmap='gist_ncar', alpha=0.1, interpolation='none')
        
    #     ax.set_title(f'Slice index: {slice_idx}')
    #     ax.axis('off')
    
    # # Turn off any unused subplots
    # for j in range(i + 1, len(axes.flat)):
    #     axes.flat[j].axis('off')

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig(f"raw_labeled_crop_2D.jpeg", bbox_inches='tight')
    # print("Figure successfully saved.")