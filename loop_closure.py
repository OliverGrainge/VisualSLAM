import faiss
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typint import Union


class LoopClosure:
    def __init__(self, threshold=0.90):
        self.threshold = threshold

        self.model = model = torch.hub.load(
            "gmberton/eigenplaces",
            "get_trained_model",
            backbone="ResNet50",
            fc_output_dim=2048,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.model.to(self.device)

        self.preprocess = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((480, 640), antialias=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.index = faiss.IndexFlatL2(2048)

    def __call__(self, image: Image.Image) -> Union[int, None]:
        feature = self.compute_feature(image)
        match = self.place_match(feature)
        return match


    def compute_feature(self, image: Image.Image) -> np.ndarray:
        image_npy = np.array(image)
        if image_npy.ndim == 2:
            image = np.stack([image_npy, image_npy, image_npy])
            image = image.transpose(1, 2, 3).astype(np.uint8)
            image = Image.fromarray(image)

        image = self.preprocess(image)
        with torch.no_grad():
            feature = self.model(image[None, :].to(self.device)).detach().cpu().numpy()
        return feature

    def place_match(self, feature: np.ndarray) -> Union[int, None]:
        assert feature.ndim == 2
        distance, idx = self.search(feature, 1)
        distance = distance.flatten()
        idx = idx.flatten()
        if distance[0] < self.threshold: 
            return  idx[0]
        else: 
            return None

    def reset(self) -> None:
        self.index = faiss.IndexFlatL2(2048)
        


