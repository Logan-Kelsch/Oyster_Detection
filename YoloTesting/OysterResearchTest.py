import yaml
import os
from glob import glob

# Define path to dataset YAML
yaml_path = os.path.join(dataset.location, "data.yaml")

# Load class names from YAML
with open(yaml_path, "r") as f:
    data_yaml = yaml.safe_load(f)

class_names = data_yaml['names']
num_classes = len(class_names)

# Set test label directory
label_dir = os.path.join(dataset.location, "test/labels")
label_files = glob(os.path.join(label_dir, "*.txt"))

# Count instances per class
class_counts = [0] * num_classes

for label_file in label_files:
    with open(label_file, "r") as f:
        for line in f:
            if line.strip():
                class_id = int(line.split()[0])
                if 0 <= class_id < num_classes:
                    class_counts[class_id] += 1

# Print results
print("📊 Class Distribution in Test Set:")
for class_id, class_name in class_names.items():
    print(f"Class {class_id} ({class_name}): {class_counts[int(class_id)]} samples")
