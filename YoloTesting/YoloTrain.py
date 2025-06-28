import os
from ultralytics import YOLO

# Configuration (adjust as needed)
TRAIN_CONFIG = {
    "data": "REU_Oyster_2024_Improved-2/data.yaml",
    "epochs": 200,
    "imgsz": 960,
    "batch": 16,
    "patience": 20,
    "amp": True,
    "optimize": True,
    "project": "oysterTrainedModels",
    "exist_ok": True
}

# Get all .pt models in the models folder
model_dir = "models"
model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]

if not model_files:
    print(f"No .pt models found in {model_dir}!")
    exit()

print(f"Found {len(model_files)} models to train:")
for i, model_file in enumerate(model_files, 1):
    model_path = os.path.join(model_dir, model_file)
    model_name = os.path.splitext(model_file)[0]

    print(f"\n{'=' * 50}")
    print(f"TRAINING MODEL {i}/{len(model_files)}: {model_name}")
    print(f"{'=' * 50}")

    try:
        # Load model
        model = YOLO(model_path)

        # Train with unique run name
        results = model.train(
            **TRAIN_CONFIG,
            name=model_name  # Saves runs in oysterTrainedModels/model_name
        )

        print(f"\n✅ Successfully trained {model_name}")
    except Exception as e:
        print(f"\n❌ Training failed for {model_name}: {str(e)}")

print("\nAll models processed!")