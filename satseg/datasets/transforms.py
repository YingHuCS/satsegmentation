"""PyTorch-compatible transformations.
"""

import random

import torch
import numpy as np
from PIL import Image

import torchvision


# Callable to convert a RGB image into a PyTorch tensor.
ImageToTensor = torchvision.transforms.ToTensor


class MaskToTensor(object):
    """Callable to convert a PIL image into a PyTorch tensor.
    """

    def __call__(self, image):
        """Converts the image into a tensor.

        Args:
          image: the PIL image to convert into a PyTorch tensor.

        Returns:
          The converted PyTorch tensor.
        """

        return torch.from_numpy(np.array(image, dtype=np.uint8)).long()


class ConvertImageMode(object):
    """Callable to convert a PIL image into a specific image mode (e.g. RGB, P)
    """

    def __init__(self, mode):
        """Creates an `ConvertImageMode` instance.

        Args:
          mode: the PIL image mode string
        """

        self.mode = mode

    def __call__(self, image):
        """Applies to mode conversion to an image.

        Args:
          image: the PIL.Image image to transform.
        """
        image = image.convert(self.mode)

        if self.mode == 'P':
            image = np.array(image)
            image[image!=0] = 1
            image = image.astype(np.uint8)
            image = Image.fromarray(image)

        return image


class JointCompose(object):
    """Callable to transform an image and it's mask at the same time.
    """

    def __init__(self, transforms):
        """Creates an `JointCompose` instance.

        Args:
          transforms: list of tuple with (image, mask) transformations.
        """

        self.transforms = transforms

    def __call__(self, image, mask):
        """Applies multiple transformations to the images and the mask at the same time.

        Args:
          images: the PIL.Image images to transform.
          mask: the PIL.Image mask to transform.

        Returns:
          The transformed PIL.Image (images, mask) tuple.
        """

        for transform in self.transforms:
            image, mask = transform(image, mask)

        return image, mask


class JointTransform(object):
    """Callable to compose non-joint transformations into joint-transformations on images and mask.

    Note: must not be used with stateful transformations (e.g. rngs) which need to be in sync for image and mask.
    """

    def __init__(self, image_transform, mask_transform):
        """Creates an `JointTransform` instance.

        Args:
          image_transform: the transformation to run on the images or `None` for no-op.
          mask_transform: the transformation to run on the mask or `None` for no-op.

        Returns:
          The (images, mask) tuple with the transformations applied.
        """

        self.image_transform = image_transform
        self.mask_transform = mask_transform

    def __call__(self, image, mask):
        """Applies the transformations associated with images and their mask.

        Args:
          images: the PIL.Image images to transform.
          mask: the PIL.Image mask to transform.

        Returns:
          The PIL.Image (images, mask) tuple with images and mask transformed.
        """

        if self.image_transform is not None:
            image = self.image_transform(image)

        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        return image, mask


class JointRandomVerticalFlip(object):
    """Callable to randomly flip images and its mask top to bottom.
    """

    def __init__(self, p):
        """Creates an `JointRandomVerticalFlip` instance.

        Args:
          p: the probability for flipping.
        """

        self.p = p

    def __call__(self, image, mask):
        """Randomly flips images and their mask top to bottom.

        Args:
          images: the PIL.Image image to transform.
          mask: the PIL.Image mask to transform.

        Returns:
          The PIL.Image (images, mask) tuple with either images and mask flipped or none of them flipped.
        """

        if random.random() < self.p:
            return image.transpose(Image.FLIP_TOP_BOTTOM), mask.transpose(Image.FLIP_TOP_BOTTOM)
        else:
            return image, mask


class JointRandomHorizontalFlip(object):
    """Callable to randomly flip images and their mask left to right.
    """

    def __init__(self, p):
        """Creates an `JointRandomHorizontalFlip` instance.

        Args:
          p: the probability for flipping.
        """

        self.p = p

    def __call__(self, image, mask):
        """Randomly flips image and their mask left to right.

        Args:
          images: the PIL.Image images to transform.
          mask: the PIL.Image mask to transform.

        Returns:
          The PIL.Image (images, mask) tuple with either images and mask flipped or none of them flipped.
        """

        if random.random() < self.p:
            return image.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            return image, mask


class JointRandomRotation(object):
    """Callable to randomly rotate images and their mask.
    """

    def __init__(self, p, degree):
        """Creates an `JointRandomRotation` instance.

        Args:
          p: the probability for rotating.
        """

        self.p = p

        methods = {90: Image.ROTATE_90, 180: Image.ROTATE_180, 270: Image.ROTATE_270}

        if degree not in methods.keys():
            raise NotImplementedError("We only support multiple of 90 degree rotations for now")

        self.method = methods[degree]

    def __call__(self, image, mask):
        """Randomly rotates images and their mask.

        Args:
          images: the PIL.Image image to transform.
          mask: the PIL.Image mask to transform.

        Returns:
          The PIL.Image (images, mask) tuple with either images and mask rotated or none of them rotated.
        """

        if random.random() < self.p:
            return image.transpose(self.method), mask.transpose(self.method)
        else:
            return image, mask

class JointRandomResize(object):

    def __init__(self, scale_tuple):
        assert isinstance(scale_tuple, tuple)
        scale_min, scale_max = scale_tuple
        self.scale = random.uniform(scale_min, scale_max)
    
    def __call__(self, image, mask):
        w, h = image.size
        resized_w, resized_h = int(w * self.scale), int(h * self.scale)
        image = torchvision.transforms.Resize((resized_w, resized_h), interpolation=Image.BILINEAR)(image)
        mask = torchvision.transforms.Resize((resized_w, resized_h), interpolation=Image.NEAREST)(mask)

        return image, mask


class JointCenterCrop(object):

    def __init__(self, size):
        assert isinstance(size, tuple)
        self.size = size

    def __call__(self, image, mask):
        image = torchvision.transforms.CenterCrop(self.size)(image)
        mask = torchvision.transforms.CenterCrop(self.size)(mask)

        return image, mask
        

class JointRandomRotate(object):
    
    def __init__(self, degree_tuple):
        assert isinstance(degree_tuple, tuple)
        degree_min, degree_max = degree_tuple
        self.degree = random.randint(degree_min, degree_max)

    def __call__(self, image, mask):
        image = torchvision.transforms.functional.rotate(image, self.degree)
        mask = torchvision.transforms.functional.rotate(mask, self.degree)

        return image, mask 


class JointRandomCrop(object):

    def __init__(self, size):
        assert isinstance(size, int)
        self.size = size

    def __call__(self, image, mask):
        l = image.size[0]
        new_l = self.size
        
        if new_l <= l:
            top = np.random.randint(0, l - new_l)
            area = (top, top, top+new_l, top+new_l)
            image_new = image.crop(area)
            mask_new = mask.crop(area)
        else:
            image_new = Image.new('RGB', (new_l,new_l), 0)
            image_new.paste(image)
            mask_new = np.zeros((new_l,new_l)).astype(np.uint8)
            mask = np.array(mask)
            mask_new[0:l, 0:l] = mask
            mask_new = Image.fromarray(mask_new)              

        return image_new, mask_new

