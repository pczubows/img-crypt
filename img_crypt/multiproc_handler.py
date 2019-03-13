from multiprocessing import Process, Queue
from random import seed, random

import numpy as np
from PIL import Image

from cml import cml_algo_encrypt, cml_algo_decrypt


class MultiprocHandler:
    """Handler for CML encrypting image RGB channels in parallel

    For short description of CML algorithm see cml.py docstring.

    Attributes:
        queues (list): List of multiprocessing queues for each RGB color channel

    """
    def __init__(self):
        self.queues = [Queue() for _ in range(3)]

    def multiproc_encrypt(self, im, key, iterations, cycles):
        """Convert PIL image to numpy array and encrypt with cml algorithm

        Analogous to cml.cml_encrypt, with the difference of each RGB channel being
        encrypted in parallel.

        Parameters:
            im (PIL.Image): Image to be encrypted loaded into Pillow Image object
            key (float, bytes): Tuple containing key for CML encryption.
                First item is map control parameter in range from 0 to 0.5.
                Second is random generator seed
            iterations (int): number of iterations over sing for CML algorithm
            cycles (int): number of cycles for cml algorithm

        Returns:
            PIL.Image: Encrypted image
        """
        pixels = np.array(im)
        channel_size = pixels[:, :, 0].size

        dims = ()
        x, y = im.size
        dims += (y, x)
        dims += (3,)

        new_pixels = np.empty(dims, dtype=np.uint8)

        p, s = key
        seed(s)

        rand = []
        for i in range(3):
            rand.append([random() for _ in range(cycles * channel_size)])

        for i in range(3):
            proc = Process(target=self.encrypt_channel, args=(i, pixels[:, :, i], rand[i], p, iterations, cycles))
            proc.start()

        for i in range(3):
            new_pixels[:, :, i] = self.queues[i].get()

        return Image.fromarray(new_pixels)

    def multiproc_decrypt(self, im, key, iterations=25, cycles=5):
        """Convert PIL image to numpy array and decrypt with cml algorithm

        Analogous to cml.cml_decrypt, with the difference of each RGB channel being
        encrypted in parallel.

        Parameters:
            im (PIL.Image): Image to be encrypted loaded into Pillow Image object
            key (float, bytes): Tuple containing key for CML encryption.
                First item is map control parameter in range from 0 to 0.5.
                Second is random generator seed
            iterations (int): number of iterations over sing for CML algorithm
            cycles (int): number of cycles for cml algorithm

        Returns:
            PIL.Image: Encrypted image
        """
        pixels = np.array(im)
        channel_size = pixels[:, :, 0].size

        dims = ()
        x, y = im.size
        dims += (y, x)
        dims += (3,)

        new_pixels = np.empty(dims, dtype=np.uint8)

        p, s = key
        seed(s)

        rand = []

        for i in range(3):
            rand.append([random() for _ in range(cycles * channel_size)])

        for i in range(3):
            proc = Process(target=self.decrypt_channel, args=(i, pixels[:, :, i], rand[i], p, iterations, cycles))
            proc.start()

        for i in range(3):
            new_pixels[:, :, i] = self.queues[i].get()

        return Image.fromarray(new_pixels)

    def encrypt_channel(self, channel_index, channel, rand, p, iterations, cycles):
        """Encrypt single RGB color channel

        Parameters:
            channel_index (int): Index of the queue corresponding to channel in queue list
            channel (np.array): Single color channel extracted from numpy array containing image
            rand (list): List of random floating point numbers
            p (float): Chaotic map control parameter
            iterations (int): number of iterations over single pixel for CML algorithm
            cycles (int): number of cycles over whole image for cml algorithm
        """
        pixels_flat = channel.flat
        pixel_num = len(pixels_flat)

        cml_algo_encrypt(pixels_flat, pixel_num, rand, p, cycles, iterations)

        self.queues[channel_index].put(channel)

    def decrypt_channel(self, channel_index, channel, rand, p, iterations, cycles):
        """Decrypt single RGB color channel

        Parameters:
            channel_index (int): Index of the queue corresponding to channel in queue list
            channel (np.array): Single color channel extracted from numpy array containing image
            rand (list): List of random floating point numbers
            p (float): Chaotic map control parameter
            iterations (int): number of iterations over single pixel for CML algorithm
            cycles (int): number of cycles over whole image for cml algorithm
        """
        pixels_flat = channel.flat
        pixel_num = len(pixels_flat)

        cml_algo_decrypt(pixels_flat, pixel_num, rand, p, cycles, iterations)

        self.queues[channel_index].put(channel)
