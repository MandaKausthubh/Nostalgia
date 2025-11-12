from datasets.ImageNetV2 import save_split_indices

if __name__ == "__main__":
    root_dir = "./data/imagenetv2"
    save_split_indices(root=root_dir, train_ratio=0.9, split_file="imagenetv2_splits.json")