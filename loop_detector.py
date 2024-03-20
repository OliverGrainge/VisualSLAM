import faiss
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typing import Union


class LoopDetector:
    def __init__(self, threshold=0.40):
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
        if self.index.ntotal < 1:
            self.index.add(feature)
            return False, None, None
        match_sucess, match_idx, match_dist = self.place_match(feature)
        self.index.add(feature)
        return match_sucess, match_idx, match_dist

    def compute_feature(self, image: Image.Image) -> np.ndarray:
        image_npy = np.array(image)
        if image_npy.ndim == 2:
            image = np.stack([image_npy, image_npy, image_npy])
            image = image.transpose(1, 2, 0).astype(np.uint8)
            image = Image.fromarray(image)

        image = self.preprocess(image)
        with torch.no_grad():
            feature = self.model(image[None, :].to(self.device)).detach().cpu().numpy()
        return feature

    def place_match(self, feature: np.ndarray) -> Union[int, None]:
        assert feature.ndim == 2
        distance, idx = self.index.search(feature, min(self.index.ntotal, 100))
        distance = distance.flatten()
        idx = idx.flatten()
        distance_mask = (distance < self.threshold)
        distance = distance[distance_mask]
        idx = idx[distance_mask]
        idx = idx.flatten()
        locality_mask = ~np.isin(idx, np.arange(self.index.ntotal-10, self.index.ntotal+10))
        idx = idx[locality_mask]
        distance = distance[locality_mask]
        print(distance, idx)
        if len(distance) > 0:
            if distance[0] < self.threshold and idx[0] not in list(
                range(self.index.ntotal - 20, self.index.ntotal + 20)
            ):
                return True, idx[0], distance[0]
            else:
                return False, None, None
        else:
            return False, None, None

    def reset(self) -> None:
        self.index = faiss.IndexFlatL2(2048)
