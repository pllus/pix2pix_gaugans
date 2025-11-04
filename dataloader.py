import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class LandscapeDataset(Dataset):
    def __init__(self, root_dir):
        self.input_dir = os.path.join(root_dir, "input")
        self.label_dir = os.path.join(root_dir, "label")
        self.image_names = [
            f.replace("_gtFine_color.png", "")
            for f in os.listdir(self.input_dir)
            if f.endswith("_gtFine_color.png")
        ]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        name = self.image_names[index]

        input_path = os.path.join(self.input_dir, f"{name}_gtFine_color.png")
        label_path = os.path.join(self.label_dir, f"{name}_leftImg8bit.png")

        # Read both images
        input_image = np.array(Image.open(input_path).convert("RGB"))
        target_image = np.array(Image.open(label_path).convert("RGB"))

        # Convert to PyTorch tensors (C, H, W)
        input_tensor = torch.from_numpy(input_image).permute(2, 0, 1).float() / 255.0
        target_tensor = torch.from_numpy(target_image).permute(2, 0, 1).float() / 255.0

        return input_tensor, target_tensor
