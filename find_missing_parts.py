import os

def find_missing_parts(data_dir, total_parts=3000):
    """
    Finds missing 'part_x' folders in a given directory.
    """
    if not os.path.exists(data_dir):
        print(f"Error: The directory '{data_dir}' does not exist.")
        return []

    expected_indices = set(range(total_parts))
    found_indices = set()

    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        
        if os.path.isdir(item_path) and item.startswith("part_"):
            try:
                index = int(item.split("_")[1])
                found_indices.add(index)
            except ValueError:
                continue

    missing_indices = sorted(list(expected_indices - found_indices))
    return missing_indices

def format_slurm_array(indices):
    """
    Formats a list of integers into a SLURM-friendly array string.
    Compresses consecutive numbers into ranges (e.g., 1-3,5,7-9).
    """
    if not indices:
        return ""
        
    ranges = []
    start = indices[0]
    prev = indices[0]
    
    for i in indices[1:]:
        if i == prev + 1:
            prev = i
        else:
            if start == prev:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}-{prev}")
            start = i
            prev = i
            
    # Add the final range/number
    if start == prev:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}-{prev}")
        
    # Join everything with commas and no spaces
    return ",".join(ranges)

if __name__ == "__main__":
    DIRECTORY_PATH = "./data_oo_3d" 
    
    missing = find_missing_parts(DIRECTORY_PATH, total_parts=3000)
    
    print(f"Total missing folders: {len(missing)}\n")
    
    if missing:
        # 1. Simple comma-separated list (no spaces)
        simple_format = ",".join(map(str, missing))
        print("Missing indices (Simple comma-separated):")
        print(simple_format)
        print("-" * 40)
        
        # 2. Compressed range format (best for SLURM character limits)
        slurm_format = format_slurm_array(missing)
        print("Missing indices (SLURM range-compressed):")
        print(slurm_format)