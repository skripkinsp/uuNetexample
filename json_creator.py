import json
import os
output_dir = "nnUNet_raw/Dataset001_MyTask"
task_name = "Dataset001_MyTask"
num_training = len(os.listdir(f"{output_dir}/imagesTr"))
print(num_training)
dataset_info = {
    "labels": {
        "0": "background",
        "1": "object"
    },
    "channel_names": {"0": "MRI"},
    "numTraining": num_training,
    "file_ending": ".nii.gz"
}

with open(f"{output_dir}/dataset.json", "w") as f:
    json.dump(dataset_info, f, indent=4)