import os
import io
import json
import random
import torch
import numpy as np
import logging
from typing import Optional, Sequence, Union, Dict

import datetime
from pytz import timezone, utc


class TextBatchLoader:

    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.n_iter = int(len(data) / batch_size)
        if len(data) % batch_size != 0:
            self.n_iter += 1

    def __len__(self):
        return self.n_iter

    def __iter__(self):
        for idx in range(self.n_iter):
            indices = list(range(idx * self.batch_size,
                                 (idx + 1) * self.batch_size))
            batch = self.data[idx * self.batch_size:(idx + 1) * self.batch_size]
            yield indices, batch


def get_logger(name=None):
    if not name:
        name = 'main'
        
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    def customTime(*args):
        utc_dt = datetime.datetime.now()
        my_tz = timezone("Asia/Seoul")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()
    logging.Formatter.converter = customTime
    return logger


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 0:
        torch.cuda.manual_seed_all(seed)
        
        
def _make_w_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f_dirname = os.path.dirname(f)
        if f_dirname != "":
            os.makedirs(f_dirname, exist_ok=True)
        f = open(f, mode=mode)
    return f


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f


def jdump(obj, f, mode="w", indent=4, default=str):
    """Dump a str or dictionary to a file in json format.

    Args:
        obj: An object to be written.
        f: A string path to the location on disk.
        mode: Mode for opening the file.
        indent: Indent for storing json dictionaries.
        default: A function to handle non-serializable entries; defaults to `str`.
    """
    f = _make_w_io_base(f, mode)
    if isinstance(obj, (dict, list)):
        json.dump(obj, f, indent=indent, default=default)
    elif isinstance(obj, str):
        f.write(obj)
    else:
        raise ValueError(f"Unexpected type: {type(obj)}")
    f.close()


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict
