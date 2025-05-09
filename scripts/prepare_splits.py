import os
import random

def split_train_val(input_path, train_path, val_path, val_ratio=0.2, seed=42):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"{input_path} not found")

    with open(input_path, 'r') as f:
        all_files = [line.strip() for line in f.readlines()]

    random.seed(seed)
    random.shuffle(all_files)

    split_idx = int(len(all_files) * (1 - val_ratio))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    with open(train_path, 'w') as f:
        for item in train_files:
            f.write(item + '\n')

    with open(val_path, 'w') as f:
        for item in val_files:
            f.write(item + '\n')

    print(f"✅ Split complete: {len(train_files)} train / {len(val_files)} val")

if __name__ == "__main__":
    base_path = "data/xray_images/"
    input_list = os.path.join(base_path, "train_val_list.txt")
    train_out = os.path.join(base_path, "train.txt")
    val_out = os.path.join(base_path, "val.txt")

    split_train_val(input_list, train_out, val_out, val_ratio=0.2)