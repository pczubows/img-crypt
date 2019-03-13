from multiprocessing import Process, Queue
from random import seed, random

import numpy as np
from PIL import Image

from cml import cml_algo_encrypt, cml_algo_decrypt


class MultiprocHandler:
    def __init__(self):
        self.queues = [Queue() for _ in range(3)]

    def multiproc_encrypt(self, im, key, iterations=25, cycles=5):
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
        pixels_flat = channel.flat
        pixel_num = len(pixels_flat)

        cml_algo_encrypt(pixels_flat, pixel_num, rand, p, cycles, iterations)

        self.queues[channel_index].put(channel)

    def decrypt_channel(self, channel_index, channel, rand, p, iterations, cycles):
        pixels_flat = channel.flat
        pixel_num = len(pixels_flat)

        cml_algo_decrypt(pixels_flat, pixel_num, rand, p, cycles, iterations)

        self.queues[channel_index].put(channel)
