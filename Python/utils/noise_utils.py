from abc import abstractmethod

import cv2 as cv
import numpy as np
from evolution.base.base_strategies import Strategy


class NoiseStrategy(Strategy):
    @abstractmethod
    def generate_noise(self, image: np.array) -> np.array:
        raise NotImplementedError

    @abstractmethod
    def get_value(self):
        raise NotImplementedError


class NoNoise(NoiseStrategy):
    def generate_noise(self, image: np.array) -> np.array:
        noise = np.zeros_like(image)
        return noise

    def printable_identifier(self):
        return "no_noise"

    def get_value(self):
        return 0


class SaltNoise(NoiseStrategy):
    def __init__(self, amount: float = 0.004) -> None:
        super().__init__()
        self.amount = amount

    def generate_noise(self, image: np.array) -> np.array:
        """Creates salty noise with a given amount

        Args:
            image (np.array): The image will be used to detive shape and calculate the absolute amount and the max noise value

        Returns:
            np.array: Noise image with the specified amount of salt of same type as the input image
        """
        n_salt = np.ceil(self.amount * image.size)

        noise = np.zeros_like(image)
        salt_value = np.iinfo(image.dtype).max

        coordinates = tuple([np.random.randint(0, i - 1, int(n_salt)) for i in image.shape])
        noise[coordinates] = salt_value
        return noise

    def printable_identifier(self):
        return "salt"

    def get_value(self):
        return self.amount


class HLinesNoise(NoiseStrategy):
    def __init__(self, spacing: int = 32) -> None:
        self.spacing = spacing

    def generate_noise(self, image: np.array) -> np.array:
        """Creates horizontal noise with a given spacing

        Args:
            image (np.array): The image will be used to derive shape and the max noise value

        Returns:
            np.array: Noise image with the specified horizontal lines of same type as the input image
        """
        max_value = np.iinfo(image.dtype).max
        noise = np.zeros_like(image)
        noise[:: self.spacing] = max_value
        return noise

    def printable_identifier(self):
        return "hlines"

    def get_value(self):
        return self.spacing


class VLinesNoise(NoiseStrategy):
    def __init__(self, spacing: int = 32) -> None:
        self.spacing = spacing

    def generate_noise(self, image: np.array) -> np.array:
        """Creates vertical noise with a given spacing

        Args:
            image (np.array): The image will be used to derive shape and the max noise value

        Returns:
            np.array: Noise image with the specified vertical lines of same type as the input image
        """
        max_value = np.iinfo(image.dtype).max
        noise = np.zeros_like(image)
        noise[:, :: self.spacing] = max_value
        return noise

    def printable_identifier(self):
        return "vlines"

    def get_value(self):
        return self.spacing


def _rotate_image(image: np.array, angle: float) -> np.array:
    """Rotates the image by a given angle

    Args:
        image (np.array): The image to rotate
        angle (float): The rotation angle in degree

    Returns:
        np.array: The rotated image
    """
    ic = tuple(np.array(image.shape[1::-1]) / 2)
    R = cv.getRotationMatrix2D(ic, angle, 1.0)
    return cv.warpAffine(image, R, image.shape[1::-1], flags=cv.INTER_LINEAR | cv.WARP_INVERSE_MAP)


class GridNoise(NoiseStrategy):
    def __init__(self, hspacing: int = 32, vspacing: int = 32, angle: float = 0) -> None:
        self.hspacing = hspacing
        self.vspacing = vspacing

        self.hlines_noise_strategy = HLinesNoise(hspacing)
        self.vlines_noise_strategy = VLinesNoise(vspacing)
        self.angle = angle

    def generate_noise(self, image: np.array, **kwargs) -> np.array:
        """Creates rotated grid noise with a given angle and vertical and horizontal spacing.

        Args:
            image (np.array): The image will be used to derive the max noise value

        Returns:
            np.array: The grid noise image
        """
        h, w = image.shape[:2]
        border = int(np.sqrt(w ** 2 + h ** 2) - h)
        border_half = border // 2
        expanded_image = cv.copyMakeBorder(image, border, 0, border, 0, cv.BORDER_CONSTANT)
        # expanded_image = image

        h_noise = self.hlines_noise_strategy.generate_noise(expanded_image)
        v_noise = self.vlines_noise_strategy.generate_noise(expanded_image)

        noise = h_noise + v_noise
        return _rotate_image(noise, self.angle)[
               border_half: (h + border_half), border_half: (w + border_half)
               ]

    def printable_identifier(self):
        return "straight_grid" if self.angle == 0 else "angled_grid"

    def get_value(self):
        return self.hlines_noise_strategy.get_value()


def add_noise_to_image(original_image: np.array, noise_image: np.array) -> np.array:
    """Adds a noise image to an original image

    Adds the two images and applies a threshold to the result.

    Args:
        original_image (np.array): The "original" image
        noise_image (np.array): The noise image

    Returns:
        np.array: The combination of both images as 8-Bit image
    """
    noisy_edge_image = original_image + noise_image
    noisy_edge_image = cv.threshold(noisy_edge_image, 0, 255, cv.THRESH_BINARY)[1].astype(np.uint8)
    return noisy_edge_image
