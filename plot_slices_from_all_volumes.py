import os
import re
import zarr
import argparse
import numpy as np
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt

# ---------------- CONFIGURATION ------------------------------------------------------------
LOWER_PERCENTILE = 1.0                 
UPPER_PERCENTILE = 99.0                
TARGET_SIZE = (2048, 2048)
OUTPUT_DIR = "volumes"
VOLUMES_TO_BE_INVERTED = [
    "jrc_ccl81-covid-1", "jrc_fly-acc-calyx-1", "jrc_fly-fsb-1", "jrc_hela-4", "jrc_hela-22",
    "jrc_hela-h89-1", "jrc_hela-h89-2", "jrc_hela-nz-1", "jrc_hela-nz-2", "jrc_mus-cerebellum-4",
    "jrc_mus-cerebellum-5", "jrc_mus-cortex-3", "jrc_mus-dorsal-striatum-2", "jrc_mus-dorsal-striatum",
    "jrc_mus-granule-neurons-1", "jrc_mus-granule-neurons-2", "jrc_mus-granule-neurons-3",
    "jrc_mus-hippocampus-2", "jrc_mus-hippocampus-3", "jrc_mus-nacc-2", "jrc_mus-nacc-3",
    "jrc_mus-nacc-4", "jrc_mus-pancreas-3"
] 
# -------------------------------------------------------------------------------------------

def get_recon_sort_key(recon_name):
    match = re.search(r'recon-(\d+)', recon_name)
    return int(match.group(1)) if match else float('inf')

def get_em_subfolder_sort_key(folder_name):
    if folder_name.endswith('-uint8'): return 0
    elif folder_name.endswith('-uint8_1'): return 1
    elif folder_name.endswith('-uint16'): return 2
    elif folder_name.endswith('-int16'): return 3
    return 4

def normalize_to_uint8(slice_2d, volume_name):
    """Safely casts heterogenous EM arrays to standard 8-bit grayscale with a black background."""
    
    # Drop completely uniform slices early
    if slice_2d.min() == slice_2d.max():
        return None
        
    # Convert to float for safe math
    slice_float = slice_2d.astype(np.float32)
    
    # Identify background padding values dynamically
    bg_black = slice_float.min()
    bg_white = slice_float.max()
    
    # Create a 1D array of only the valid tissue pixels
    valid_mask = (slice_float != bg_black) & (slice_float != bg_white)
    valid_pixels = slice_float[valid_mask]
    
    # Fallback if the slice is entirely padding
    if valid_pixels.size == 0:
        return None
    
    # Calculate percentiles ONLY on the valid tissue pixels
    p_low, p_high = np.percentile(valid_pixels, (LOWER_PERCENTILE, UPPER_PERCENTILE))

    # Fallback if percentiles are identical
    if p_high - p_low == 0:
        p_low, p_high = valid_pixels.min(), valid_pixels.max()
        if p_high - p_low == 0:
            return None

    # Normalize the ENTIRE slice (including padding) to 0.0 - 1.0 range
    normalized = np.clip((slice_float - p_low) / (p_high - p_low), 0, 1)
    
    # Invert brightness 
    if volume_name in VOLUMES_TO_BE_INVERTED:
        normalized = 1.0 - normalized
    
    # Scale to 255 and cast back to uint8
    return (normalized * 255.0).astype(np.uint8)

def extract_and_preprocess(current_array, axis, slice_idx, volume_name):
    """Extracts, crops, normalizes, and resizes a single 2D slice."""
    # 0:Z, 1:Y, 2:X
    if axis == 0:
        slice_2d = current_array[slice_idx, :, :]
    elif axis == 1:
        slice_2d = current_array[:, slice_idx, :]
    elif axis == 2:
        slice_2d = current_array[:, :, slice_idx]
    else:
        raise ValueError("Invalid axis")

    slice_2d = np.array(slice_2d)

    # --- NORMALIZE ---
    slice_2d_uint8 = normalize_to_uint8(slice_2d, volume_name)

    # --- CROP TO MINIMAL BOUNDING BOX---
    is_foreground = (slice_2d_uint8 != 0) & (slice_2d_uint8 != 255)
    rows = np.any(is_foreground, axis=1)
    cols = np.any(is_foreground, axis=0)
            
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    slice_2d_uint8 = slice_2d_uint8[rmin:rmax+1, cmin:cmax+1]

    # --- RESIZE ---
    img = Image.fromarray(slice_2d_uint8).resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    
    return np.array(img)

def main():
    parser = argparse.ArgumentParser(description="Generate 3x3 QC grids for EM volumes.")
    parser.add_argument("--root_directory", type=str, default="/lustre/blizzard/stf218/scratch/emin/seg3d/data", help="Root directory for zarr volumes")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    zarr_paths = sorted(glob(os.path.join(args.root_directory, '*/*.zarr')))

    print(f"Found {len(zarr_paths)} Zarr volumes. Generating QC grids...")

    for zarr_path in zarr_paths:
        dataset_name = os.path.basename(zarr_path).replace('.zarr', '')
        
        try:
            zarr_root = zarr.open(zarr_path, mode='r')
        except Exception:
            print(f"Skipping {dataset_name} - Cannot open Zarr.")
            continue

        # Navigate metadata identical to the original script
        recon_keys = [k for k in zarr_root.keys() if k.startswith('recon-')]
        if not recon_keys: continue
        earliest_recon = sorted(recon_keys, key=get_recon_sort_key)[0]
        
        em_path = f"{earliest_recon}/em"
        if em_path not in zarr_root: continue

        em_subfolders = list(zarr_root[em_path].keys())
        if not em_subfolders: continue
        best_em = sorted(em_subfolders, key=get_em_subfolder_sort_key)[0]
        
        s0_path = f"{em_path}/{best_em}/s0"
        if s0_path not in zarr_root: continue

        print(f"Processing {dataset_name}...")
        current_array = zarr_root[s0_path]
        shape = current_array.shape # (Z, Y, X)

        # Get slices at 25%, 50%, and 75% for each axis
        z_slices = [shape[0]//4, shape[0]//2, 3*shape[0]//4]
        y_slices = [shape[1]//4, shape[1]//2, 3*shape[1]//4]
        x_slices = [shape[2]//4, shape[2]//2, 3*shape[2]//4]

        # Order for 3x3 grid: Row 1 (Z), Row 2 (Y), Row 3 (X)
        tasks = [
            (0, z_slices[0]), (0, z_slices[1]), (0, z_slices[2]),
            (1, y_slices[0]), (1, y_slices[1]), (1, y_slices[2]),
            (2, x_slices[0]), (2, x_slices[1]), (2, x_slices[2])
        ]

        # Set up matplotlib figure with zero spacing
        fig, axes = plt.subplots(3, 3, figsize=(15, 15), gridspec_kw={'wspace': 0, 'hspace': 0})
        axes = axes.flatten()

        for idx, (axis, slice_idx) in enumerate(tasks):
            processed_img = extract_and_preprocess(current_array, axis, slice_idx, dataset_name)
            axes[idx].imshow(processed_img, cmap='gray', aspect='auto')
            axes[idx].axis('off') # Remove ticks and borders

        # Save as JPEG directly into the volumes folder
        output_filepath = os.path.join(OUTPUT_DIR, f"{dataset_name}.jpg")
        plt.savefig(output_filepath, bbox_inches='tight', pad_inches=0, format='jpg', dpi=300)
        plt.close(fig)

    print(f"Finished! QC images saved to the '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()