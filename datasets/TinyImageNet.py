import os
from glob import glob
from .baseDataset import BaseDataset

class TinyImageNetDataset(BaseDataset):
    def __init__(self, root: str, split: str = "train", transform=None):
        super().__init__(root=root, transform=transform, name=f"TinyImageNet-{split}", split=split)
        self._load_metadata()
        self._load_split()

    def _load_metadata(self):
        wnid_file = os.path.join(self.root, "wnids.txt")
        if not os.path.exists(wnid_file):
            raise FileNotFoundError(f"Missing wnids.txt at {wnid_file}")
        with open(wnid_file, "r") as f:
            wnids = [line.strip() for line in f if line.strip()]
        self._set_classes(wnids)

    def _load_split(self):
        if self.split in ["train", "val"]:
            samples = self._load_train_samples()
            train_samples, val_samples = self._train_val_split(samples)
            if self.split == "train":
                self.samples = train_samples
            else:
                self.samples = val_samples

        elif self.split == "test":
            self.samples = self._load_val_samples()  # TinyImageNet "val" folder acts as test
        else:
            raise ValueError(f"Unknown split: {self.split}")

    def _load_train_samples(self):
        samples = []
        train_dir = os.path.join(self.root, "train")
        for wnid in self.classes:
            img_dir = os.path.join(train_dir, wnid, "images")
            imgs = glob(os.path.join(img_dir, "*.JPEG")) + glob(os.path.join(img_dir, "*.jpg"))
            label = self.class_to_idx[wnid]
            samples.extend([(img, label) for img in imgs])
        return samples

    def _load_val_samples(self):
        val_dir = os.path.join(self.root, "val", "images")
        val_annotations = os.path.join(self.root, "val", "val_annotations.txt")
        if not os.path.exists(val_annotations):
            raise FileNotFoundError(f"Missing val_annotations.txt at {val_annotations}")

        img_to_label = {}
        with open(val_annotations, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    fname, wnid = parts[0], parts[1]
                    img_to_label[fname] = self.class_to_idx[wnid]

        samples = []
        for img_path in glob(os.path.join(val_dir, "*.JPEG")):
            fname = os.path.basename(img_path)
            if fname in img_to_label:
                label = img_to_label[fname]
                samples.append((img_path, label))
        return samples
