import zarr
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def print_and_plot_shape(base_data_dir, volume_name, n_slices=5, max_plot_dim=2000):
    """Reads uniformly spaced z-slices, prints shape, and saves a JPEG plot."""

    em_dir = Path(base_data_dir) / volume_name / f"{volume_name}.zarr" / "recon-1" / "em"
    
    # Find directories matching either pattern
    modality_dirs = list(em_dir.glob("fibsem-*")) + list(em_dir.glob("tem-*"))
    
    if not modality_dirs:
        print(f"Could not find any 'fibsem-*' or 'tem-*' directory in {em_dir}")
        return
        
    zarr_path = modality_dirs[0] / "s0"

    try:
        dataset = zarr.open(str(zarr_path), mode='r')
    except Exception as e:
        print(f"Failed to open {volume_name}: {e}")
        return

    print(f"Shape of {volume_name} = {dataset.shape} ({modality_dirs[0]})")

    # --- Plotting Logic ---
    total_z = dataset.shape[0]
    y_dim = dataset.shape[1]
    x_dim = dataset.shape[2]
    
    # Calculate uniformly spaced Z-indices
    z_indices = np.linspace(0, total_z - 1, n_slices, dtype=int)
    
    # Dynamic downsampling: Ensures no image is loaded wider than max_plot_dim
    largest_spatial_dim = max(y_dim, x_dim)
    step = max(1, largest_spatial_dim // max_plot_dim)
    
    # Setup the matplotlib figure
    fig, axes = plt.subplots(1, n_slices, figsize=(4 * n_slices, 4))
    if n_slices == 1:
        axes = [axes] # Handle single slice case safely
        
    for i, z in enumerate(z_indices):
        # Extract the downsampled slice directly from disk
        slice_data = dataset[z, ::step, ::step]
        
        axes[i].imshow(slice_data, cmap='gray')
        axes[i].set_title(f"Z = {z}")
        axes[i].axis('off')
        
    plt.suptitle(f"{volume_name} (Shape: {dataset.shape})", fontsize=14)
    plt.tight_layout()
    
    output_filename = f"{volume_name}_slices.jpeg"
    plt.savefig(output_filename, format='jpeg', dpi=150, bbox_inches='tight')
    plt.close() # CRITICAL: Free memory after saving
    print(f"  -> Saved {output_filename}")

# --- Main Execution ---

BASE_DATA_DIR = "data"
VOLUME_NAMES = [
    # 'jrc_hela-1', 'jrc_hela-21', 'jrc_hela-h89-1', 'jrc_hela-h89-2', 'jrc_mus-pancreas-3', # <100K files
    # 'jrc_hela-2', 'jrc_hela-3', 'jrc_hela-4', 'jrc_hela-22', 'jrc_hela-bfa', 'jrc_jurkat-1', 'jrc_macrophage-2', 'jrc_ccl81-covid-1', 'jrc_cos7-11', # 100K-1M
    # 'jrc_ctl-id8-1', 'jrc_ctl-id8-2', 'jrc_ctl-id8-3', 'jrc_ctl-id8-4', 'jrc_ctl-id8-5', 'jrc_mus-pancreas-2', 'jrc_mus-sc-zp104a', 'jrc_mus-sc-zp105a', 'jrc_sum159-1', # 100K-1M
    # 'jrc_mus-kidney', 'jrc_mus-liver', 'jrc_choroid-plexus-2', 'jrc_fly-acc-calyx-1', 'jrc_fly-fsb-1', 'jrc_dauer-larva', # 1M-10M
]

for volume_name in VOLUME_NAMES:
    print_and_plot_shape(BASE_DATA_DIR, volume_name, n_slices=5)