import faiss
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from typing import Union, Tuple


class LoopDetector:
    """
    A loop detector class that utilizes a deep learning model to compute features of images and performs loop detection
    by matching these features using the FAISS library. Loop detection identifies if a current image has been seen before.

    Attributes:
        threshold (float): The distance threshold for considering two images as a match. Lower values mean more strict matching.
        model (torch.nn.Module): The deep learning model used to compute image features.
        device (str): The device on which the model will be run ('cuda' or 'cpu').
        preprocess (torchvision.transforms.Compose): The preprocessing pipeline applied to images before feature extraction.
        index (faiss.IndexFlatL2): The FAISS index for efficient similarity search of image features.

    Parameters:
        threshold (float, optional): The threshold for matching features. Defaults to 0.40.
    """
    def __init__(self, threshold=0.40):
        """
        Initializes the loop detector with a specified threshold for feature matching, loads the model, and prepares the FAISS index.
        """
        self.threshold = threshold

        self.model = torch.hub.load(
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

    def __call__(self, image: Image.Image) -> Tuple[bool, int, float]:
        """
        Processes an image through the loop detector to determine if it matches a previously seen image.

        Parameters:
            image (Image.Image): The image to process and match.

        Returns:
            Tuple[bool, Optional[int], Optional[float]]: A tuple containing a boolean indicating a successful match,
            the index of the matched image if a match is found, and the distance to the matched image. Returns (False, None, None) if no match is found.
        """
        feature = self.compute_feature(image)
        if self.index.ntotal < 1:
            self.index.add(feature)
            return False, None, None
        match_sucess, match_idx, match_dist = self.place_match(feature)
        self.index.add(feature)
        return match_sucess, match_idx, match_dist

    def compute_feature(self, image: Image.Image) -> np.ndarray:
        """
        Computes the feature vector of an image using the deep learning model.

        Parameters:
            image (Image.Image): The image to compute features for.

        Returns:
            np.ndarray: The computed feature vector.
        """
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
        """
        Attempts to match a feature vector with previously seen images.

        Parameters:
            feature (np.ndarray): The feature vector to match.

        Returns:
            Tuple[bool, Optional[int], Optional[float]]: A tuple indicating whether a match was found,
            the index of the matched image, and the distance of the match. If no match is found, returns (False, None, None).
        """
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
        """
        Resets the FAISS index, effectively clearing the memory of previously seen images.
        """
        self.index = faiss.IndexFlatL2(2048)
