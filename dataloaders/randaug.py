import mmcv
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


PARAMETER_MAX = 10


def int_parameter(level, maxval, max_level=None):
    if max_level is None:
        max_level = PARAMETER_MAX
    return int(level * maxval / max_level)


def float_parameter(level, maxval, max_level=None):
    if max_level is None:
        max_level = PARAMETER_MAX
    return float(level) * maxval / max_level


class RandAug(object):
    """refer to https://github.com/google-research/ssl_detection/blob/00d52272f
    61b56eade8d5ace18213cba6c74f6d8/detection/utils/augmentation.py#L240."""

    def __init__(
        self,
        prob: float = 1.0,
        magnitude: int = 10,
        random_magnitude: bool = True,
        magnitude_limit: int = 10,
    ):
        assert 0 <= prob <= 1, f"probability should be in (0,1) but get {prob}"
        assert (
            magnitude <= PARAMETER_MAX
        ), f"magnitude should be small than max value {PARAMETER_MAX} but get {magnitude}"

        self.prob = prob
        self.magnitude = magnitude
        self.magnitude_limit = magnitude_limit
        self.random_magnitude = random_magnitude
        self.buffer = None

    def __call__(self, imgpil):
        if np.random.random() < self.prob:
            magnitude = self.magnitude
            if self.random_magnitude:
                magnitude = np.random.randint(1, magnitude)
            imgpil = self.apply(imgpil, magnitude)
        return imgpil

    def apply(self, results, magnitude= None):
        raise NotImplementedError()

    def __repr__(self):
        return f"{self.__class__.__name__}(prob={self.prob},magnitude={self.magnitude},max_magnitude={self.magnitude_limit},random_magnitude={self.random_magnitude})"


class Identity(RandAug):
    def apply(self, imgpil, magnitude=None):
        return imgpil

class AutoContrast(RandAug):
    def apply(self, imgpil, magnitude=None):
        imgpil=ImageOps.autocontrast(imgpil)
        return imgpil


class RandEqualize(RandAug):
    def apply(self, imgpil, magnitude=None):
        imgpil=ImageOps.equalize(imgpil)
        return imgpil


class RandSolarize(RandAug):
    def apply(self, imgpil, magnitude=None):
        img=np.asarray(imgpil)
        img=mmcv.solarize(img, min(int_parameter(magnitude, 256, self.magnitude_limit), 255))
        imgpil=Image.fromarray(img)
        return imgpil


def _enhancer_impl(enhancer):
    """Sets level to be between 0.1 and 1.8 for ImageEnhance transforms of
    PIL."""

    def impl(pil_img, level, max_level=None):
        v = float_parameter(level, 1.8, max_level) + 0.1  # going to 0 just destroys it
        return enhancer(pil_img).enhance(v)

    return impl


class RandEnhance(RandAug):
    op = None

    def apply(self, imgpil, magnitude=None):
        imgpil=_enhancer_impl(self.op)(imgpil, magnitude, self.magnitude_limit)
        return imgpil


class RandColor(RandEnhance):
    op = ImageEnhance.Color


class RandContrast(RandEnhance):
    op = ImageEnhance.Contrast


class RandBrightness(RandEnhance):
    op = ImageEnhance.Brightness


class RandSharpness(RandEnhance):
    op = ImageEnhance.Sharpness

class RandPosterize(RandAug):
    def apply(self, imgpil, magnitude=None):
        magnitude = int_parameter(magnitude, 4, self.magnitude_limit)
        imgpil=ImageOps.posterize(imgpil, 4 - magnitude)
        return imgpil


class OneOf(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgpil):
        choosetrans = np.random.choice(self.transforms)
        return choosetrans(imgpil)

