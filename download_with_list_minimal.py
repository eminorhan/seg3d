import s3fs
import os
import time
import concurrent.futures
import re 

# --- Configuration ---
MAX_WORKERS = 8  # Maximum number of simultaneous downloads
MAX_DOWNLOAD_ATTEMPTS = 5  # Maximum number of times to try downloading a single dataset
BUCKET_NAME = 'janelia-cosem-datasets'

BUCKET_0_100K = [
    'cam_hum-airway-14500', 'cam_hum-airway-14771-b', 'jrc_fly-ellipsoid-body', 'jrc_fly-fsb-2', 
    'jrc_fly-protocerebral-bridge', 'jrc_hela-1', 'jrc_hela-21', 'jrc_hela-h89-1', 'jrc_hela-h89-2', 
    'jrc_mus-cerebellum-4', 'jrc_mus-cerebellum-5', 'jrc_mus-dorsal-striatum', 'jrc_mus-epididymis-1', 
    'jrc_mus-epididymis-2', 'jrc_mus-granule-neurons-2', 'jrc_mus-nacc-1', 'jrc_mus-nacc-2', 'jrc_mus-nacc-3', 
    'jrc_mus-nacc-4', 'jrc_mus-pancreas-1', 'jrc_mus-pancreas-3'
    ]  # 21

BUCKET_100K_1M_0 = [
    'aic_desmosome-1', 'aic_desmosome-2', 'aic_desmosome-3', 'jrc_ccl81-covid-1', 'jrc_cos7-11', 'jrc_cos7-1a', 
    'jrc_cos7-1b', 'jrc_ctl-id8-1', 'jrc_ctl-id8-2', 'jrc_ctl-id8-3', 'jrc_ctl-id8-4', 'jrc_ctl-id8-5', 'jrc_fly-vnc-1', 
    'jrc_hela-2', 'jrc_hela-22', 'jrc_hela-3', 'jrc_hela-4', 'jrc_hela-bfa'
    ]  # 18
    
BUCKET_100K_1M_1 = [
    'jrc_hela-nz-1', 'jrc_hela-nz-2', 'jrc_jurkat-1', 'jrc_macrophage-2', 'jrc_mus-choroid-plexus-3', 'jrc_mus-cortex-3', 'jrc_mus-dorsal-striatum-2', 
    'jrc_mus-granule-neurons-1', 'jrc_mus-granule-neurons-3', 'jrc_mus-hippocampus-2', 'jrc_mus-hippocampus-3', 'jrc_mus-kidney-glomerulus-2', 
    'jrc_mus-pancreas-2', 'jrc_mus-sc-zp104a', 'jrc_mus-sc-zp105a', 'jrc_sum159-1', 'jrc_sum159-4'
    ]  # 17

BUCKET_1M_10M_0 = [
    'jrc_choroid-plexus-2', 'jrc_dauer-larva', 'jrc_fly-acc-calyx-1', 'jrc_fly-fsb-1', 'jrc_fly-mb-1a', 'jrc_hum-airway-14953vc',
    'jrc_mus-heart-1', 'jrc_mus-hippocampus-1', 'jrc_mus-kidney-2', 'jrc_mus-kidney-3', 'jrc_mus-kidney', 'jrc_mus-liver-2',
    ]  # 12 (full)

BUCKET_1M_10M_1 = [
    'jrc_mus-liver-3', 'jrc_mus-liver-4', 'jrc_mus-liver-5', 'jrc_mus-liver-6', 'jrc_mus-liver-7', 'jrc_mus-liver', 
    'jrc_mus-meissner-corpuscle-1', 'jrc_mus-pancreas-4', 'jrc_mus-salivary-1', 'jrc_mus-skin-1', 'jrc_mus-thymus-1', 'jrc_ut21-1413-003'
    ]  # 12 (full)

BUCKET_10M_INF = [
    'jrc_fly-larva-1', 'jrc_fly-mb-z0419-20', 'jrc_mus-guard-hair-follicle', 'jrc_mus-liver-zon-1', 'jrc_mus-liver-zon-2', 
    'jrc_mus-meissner-corpuscle-2', 'jrc_mus-pacinian-corpuscle', 'jrc_zf-cardiac-1'
    ]  # 8

BUCKET_TO_DOWNLOAD = BUCKET_0_100K


def download_zarr_archive(s3_prefix_path, s3_filesystem, root_dir):
    dataset_name = s3_prefix_path.split('/')[-1]
    s3_source_path = f"{s3_prefix_path}/{dataset_name}.zarr"
    local_dest_path = os.path.join(root_dir, dataset_name, f"{dataset_name}.zarr")

    if not s3_filesystem.exists(s3_source_path):
        return f"🟡 Skipped:   {s3_source_path} does not exist."

    print(f"-> Verifying files for {dataset_name}...")
    try:
        remote_file_details = s3_filesystem.find(s3_source_path, detail=True)
    except Exception as e:
        return f"❌ Failed:    Could not list files in {s3_source_path}. Error: {e}"
        
    all_remote_files = list(remote_file_details.values())
    if not all_remote_files:
        return f"🟡 Skipped:   No files found in {s3_source_path}."

    # --- START OF DYNAMIC SELECTION & FILTERING LOGIC ---
    
    # 1. Discover all reconstructions that actually contain 'em' data
    valid_recons = set()
    for f in all_remote_files:
        rel_path = f['name'][len(s3_source_path):].lstrip('/')
        parts = rel_path.split('/')
        if len(parts) > 2 and parts[1] == 'em':
            valid_recons.add(parts[0])
            
    if not valid_recons:
        return f"🟡 Skipped:   No 'em' data found for {dataset_name}."

    # 2. Select the earliest reconstruction (e.g., recon-1 over recon-2)
    def recon_sort_key(r):
        nums = re.findall(r'\d+', r)
        # If no number is found (e.g., just 'recon'), treat it as 0 so it comes first
        num = int(nums[-1]) if nums else 0
        return (num, r) 
        
    # Grab the first valid recon: [0], optionally choose the last [-1] instead
    best_recon = sorted(list(valid_recons), key=recon_sort_key)[0]

    # 3. Discover all EM formats in the chosen reconstruction
    em_versions = set()
    for f in all_remote_files:
        rel_path = f['name'][len(s3_source_path):].lstrip('/')
        parts = rel_path.split('/')
        if len(parts) > 3 and parts[0] == best_recon and parts[1] == 'em':
            em_versions.add(parts[2])

    if not em_versions:
        return f"🟡 Skipped:   No em versions found in {best_recon} for {dataset_name}."

    # 4. Select the best EM version based on your fallback rules
    def em_score(f_name):
        score = 0
        name_lower = f_name.lower()
        
        # Give highest priority to any 8-bit data (uint8)
        if 'uint8' in name_lower:
            score += 100
        # Fallback to 16-bit data (handles both unsigned and signed)
        elif 'uint16' in name_lower or 'int16' in name_lower:
            score += 50
        else:
            score += 10
            
        # Subtracting length prefers the base name over variants 
        return score - len(f_name)

    best_em = sorted(list(em_versions), key=em_score, reverse=True)[0]
    print(f"   -> Selected version for {dataset_name}: {best_recon} / {best_em}")

    # 5. Filter the file list down to our targeted path + essential metadata
    valid_meta_dirs = {
        "",  
        best_recon,
        f"{best_recon}/em",
        f"{best_recon}/em/{best_em}",
        f"{best_recon}/em/{best_em}/s0"
    }
    
    desired_prefix = f"{best_recon}/em/{best_em}/s0/"
    filtered_remote_files = []
    
    for f_detail in all_remote_files:
        rel_path = f_detail['name'][len(s3_source_path):].lstrip('/')
        
        # Keep essential Zarr hierarchy metadata along our specific path
        if rel_path.endswith(('.zgroup', '.zattrs', '.zarray')):
            dir_name = rel_path.rsplit('/', 1)[0] if '/' in rel_path else ""
            if dir_name in valid_meta_dirs:
                filtered_remote_files.append(f_detail)
            continue
            
        # Keep actual EM 's0' data for the chosen recon and EM version
        if rel_path.startswith(desired_prefix):
            filtered_remote_files.append(f_detail)

    remote_files = filtered_remote_files
    
    if len(remote_files) <= len(valid_meta_dirs): 
        return f"🟡 Skipped:   No data chunks found in s0 for {dataset_name}."
        
    # --- END OF DYNAMIC SELECTION & FILTERING LOGIC ---

    missing_remote_files = []
    corresponding_local_files = []
    
    for remote_f_detail in remote_files:
        remote_f_path = remote_f_detail['name']
        remote_f_size = remote_f_detail['size']
        
        relative_path = remote_f_path[len(s3_source_path):].lstrip('/')
        local_f_path = os.path.join(local_dest_path, relative_path)
        
        if not os.path.exists(local_f_path) or os.path.getsize(local_f_path) != remote_f_size:
            missing_remote_files.append(remote_f_path)
            corresponding_local_files.append(local_f_path)
    
    if not missing_remote_files:
        return f"✅ Verified:  All {len(remote_files)} target files for {dataset_name} are correct."
        
    print(f"-> Found {len(remote_files)} required files for {dataset_name}. "
          f"Downloading {len(missing_remote_files)} missing or incomplete files...")

    local_dirs = {os.path.dirname(p) for p in corresponding_local_files}
    for d in local_dirs:
        os.makedirs(d, exist_ok=True)

    for attempt in range(MAX_DOWNLOAD_ATTEMPTS):
        try:
            print(f"   -> Attempt {attempt + 1}/{MAX_DOWNLOAD_ATTEMPTS} for {dataset_name}")
            s3_filesystem.get(missing_remote_files, corresponding_local_files)
            return f"✅ Success:   Downloaded {len(missing_remote_files)} files for {dataset_name}."
        except Exception as e:
            print(f"   -> Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_DOWNLOAD_ATTEMPTS - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"   -> Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
            else:
                return (f"❌ Failed:    Could not download files for {dataset_name} "
                        f"after {MAX_DOWNLOAD_ATTEMPTS} attempts.")

    return f"❌ Failed:    An unexpected error occurred with {s3_source_path}."


if __name__ == "__main__":
    s3 = s3fs.S3FileSystem(anon=True)
    local_root_dir = "data"
    os.makedirs(local_root_dir, exist_ok=True)

    print(f"Finding datasets in the bucket '{BUCKET_NAME}'...")
    try:
        all_objects = s3.ls(BUCKET_NAME, detail=True)
        all_prefix_paths = [obj['name'] for obj in all_objects if obj['type'] == 'directory']
        prefix_paths = [path for path in all_prefix_paths if path.split('/')[-1] in BUCKET_TO_DOWNLOAD]
        print(f"Found {len(prefix_paths)} total datasets to process.")
    except Exception as e:
        print(f"Could not list datasets in bucket. Error: {e}")
        prefix_paths = []

    print("-" * 50)
    print(f"Starting parallel download with up to {MAX_WORKERS} workers...")
    print("-" * 50)

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(download_zarr_archive, path, s3, local_root_dir) for path in prefix_paths]
        for future in concurrent.futures.as_completed(futures):
            status_message = future.result()
            print(status_message)

    print("-" * 50)
    print("All download attempts are complete.")