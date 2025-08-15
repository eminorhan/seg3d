import s3fs
import os
import time

import quilt3 as q3
b = q3.Bucket("s3://janelia-cosem-datasets")

# Use the ls() method to list the top-level keys
# The keys represent the datasets in this context
datasets = b.ls()

output = b.ls()

# The first item in the list of lists contains the prefixes.
prefixes_list = output[0]

# Extract and print only the prefix names.
for item in prefixes_list:
    print(item['Prefix'])

# Iterate through the example prefixes and print their contents
for prefix in prefixes_list:
    print(f"\n--- Contents of '{prefix['Prefix']}' ---")
    sub_items = b.ls(prefix['Prefix'])
    
    # Print the subdirectories and files for each dataset
    for item in sub_items[0]:
        print(f"  - Subdirectory: {item['Prefix']}")
    for item in sub_items[1]:
        print(f"  - File: {item['Key']}")

print(f"Total of {len(prefixes_list)} datasets")