import torch
from PIL import Image

class SimpleImageProcessor:
    def __init__(self):
        self.size = (224, 224)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __call__(self, image):
        """预处理图像"""
        if isinstance(image, Image.Image):
            image = image.resize(self.size)
            image_array = torch.tensor(list(image.getdata())).float()
            image_tensor = image_array.view(224, 224, 3).permute(2, 0, 1) / 255.0
        else:
            image_tensor = image

        image_tensor = (image_tensor - self.mean) / self.std
        return image_tensor

