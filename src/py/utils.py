#!/usr/bin/python3
# Author: Anja Gumpinger

import time


class timer(object):
    """Timer. """

    def __init__(self, name):
        self.name = name

    def  __enter__(self):
        self.start = time.time()

    def __exit__(self, ty, val, tb):
        end = time.time()
        print(f'{self.name}: {end - self.start:.3f} sec')