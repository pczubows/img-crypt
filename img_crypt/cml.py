"""Module responsible for encryption using chaotic map lattices

Algorithm uses chaotic map for image encryption precisely piece
wise linear chaotic map (see util.pwlcm). Pixel values converted to
floating point numbers are passed through chaotic map. Number of times
single pixel is passed through chaotic map is specified in iterations
parameter. Map's control parameter is part of encryption key.

To improve statistical properties pixel values are mixed with randomly
generated floating point numbers. The seed of random generator makes up
another part of encryption key.

The whole algorithm is repeated over image number of times specified in
cycles variable.
"""


from random import seed, random

import numpy as np
from PIL import Image

from util import pwlcm


def cml_encrypt(im, key, iterations, cycles):
    """Convert PIL image to numpy array and encrypt with cml algorithm

    Parameters:
        im (PIL.Image): Image to be encrypted loaded into Pillow Image object
        key (float, bytes): Tuple containing key for CML encryption.
            First item is map control parameter in range from 0 to 0.5.
            Second is random generator seed
        iterations (int): number of iterations over single pixel for CML algorithm
        cycles (int): number of cycles over whole image for cml algorithm

    Returns:
        PIL.Image: Encrypted image
    """
    pixels = np.array(im)
    p, s = key
    seed(s)

    for i in range(3):
        cml_encrypt_channel(pixels[:, :, i], p, iterations, cycles)

    return Image.fromarray(pixels)


def cml_decrypt(im, key, iterations, cycles):
    """Convert PIL image to numpy array and encrypt with cml algorithm

    Parameters:
        im (PIL.Image): Image to be encrypted loaded into Pillow Image object
        key (float, bytes): Tuple containing key for CML encryption.
            First item is map control parameter in range from 0 to 0.5.
            Second is random generator seed
        iterations (int): number of iterations over single pixel for CML algorithm
        cycles (int): number of cycles over whole image for cml algorithm

    Returns:
        PIL.Image: Decrypted image
    """
    pixels = np.array(im)
    p, s = key
    seed(s)

    for i in range(3):
        cml_decrypt_channel(pixels[:, :, i], p, iterations, cycles)

    return Image.fromarray(pixels)


def cml_encrypt_channel(channel, p, iterations, cycles):
    """Encrypt single RGB color channel

    Parameters:
        channel (np.array): Single color channel extracted from numpy array containing image
        p (float): Chaotic map control parameter
        iterations (int): number of iterations over single pixel for CML algorithm
        cycles (int): number of cycles over whole image for cml algorithm
    """
    pixels_flat = channel.flat
    pixel_num = len(pixels_flat)
    rand = [random() for _ in range(cycles * pixel_num)]

    cml_algo_encrypt(pixels_flat, pixel_num, rand, p, cycles, iterations)


def cml_decrypt_channel(channel, p, iterations, cycles):
    """Decrypt single RGB color channel

    Parameters:
        channel (np.array): Single color channel extracted from numpy array containing image
        p (float): Chaotic map control parameter
        iterations (int): number of iterations over single pixel for CML algorithm
        cycles (int): number of cycles over whole image for cml algorithm
    """
    pixels_flat = channel.flat
    pixel_num = len(pixels_flat)
    rand = [random() for _ in range(cycles * pixel_num)]

    cml_algo_decrypt(pixels_flat, pixel_num, rand, p, cycles, iterations)


def cml_algo_encrypt(pixels_flat, pixel_num, rand, p, cycles, iterations):
    """Apply encrypting algorithm to provided pixels

    For short description of the algorithm see the main docstring of the module

    Parameters:
        pixels_flat (numpy.flatiter): Flat iterator to numpy array containing image's pixels
        pixel_num (int): Number of pixels to be encrypted
        rand (list): List containing random floats for use in algorithm
        p (float): Chaotic map control parameter
        iterations (int): number of iterations over single pixel for CML algorithm
        cycles (int): number of cycles over whole image for cml algorithm
    """
    for cycle in range(1, cycles + 1):
        for pixel_index in range(pixel_num):

            pixel_float = pixels_flat[pixel_index - 1] / 255

            for _ in range(iterations):
                pixel_float = pwlcm(pixel_float, p)

            k = pixel_num * (cycle - 1) + pixel_index
            pixel_float = (pixel_float + rand[k]) % 1
            pixels_flat[pixel_index] = pixels_flat[pixel_index] + round(pixel_float * 255)

            if pixels_flat[pixel_index] > 255:
                pixels_flat[pixel_index] = pixels_flat[pixel_index] - 256


def cml_algo_decrypt(pixels_flat, pixel_num, rand, p, cycles, iterations):
    """Apply decrypting algorithm to provided pixels

    For short description of the algorithm see the main docstring of the module

    Parameters:
        pixels_flat (numpy.flatiter): Flat iterator to numpy array containing image's pixels
        pixel_num (int): Number of pixels to be encrypted
        rand (list): List containing random floats for use in algorithm
        p (float): Chaotic map control parameter
        iterations (int): number of iterations over single pixel for CML algorithm
        cycles (int): number of cycles over whole image for cml algorithm
    """
    for cycle in range(cycles, 0, -1):
        for pixel_index in range(pixel_num - 1, -1, -1):

            pixel_float = pixels_flat[pixel_index - 1] / 255

            for _ in range(iterations):
                pixel_float = pwlcm(pixel_float, p)

            k = pixel_num * (cycle - 1) + pixel_index
            pixel_float = (pixel_float + rand[k]) % 1
            pixels_flat[pixel_index] = pixels_flat[pixel_index] - round(pixel_float * 255)

            if pixels_flat[pixel_index] < 0:
                pixels_flat[pixel_index] = pixels_flat[pixel_index] + 256
