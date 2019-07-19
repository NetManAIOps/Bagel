import datetime
import time

__TIC_TIME = None


def tic():
    global __TIC_TIME
    __TIC_TIME = time.time()


def toc():
    return time.time() - __TIC_TIME
