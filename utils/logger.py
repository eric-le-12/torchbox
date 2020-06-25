import os
from datetime import datetime
import logging

LOG_DIR = "saved/logs"


def make_file(sess_name, time_str):
    file_name = sess_name + "_" + time_str + ".txt"
    f = open(os.path.join(LOG_DIR, file_name), "w+")
    f.close()
    return os.path.join(LOG_DIR, file_name)


def log_initilize(log_path):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to info
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    handler.terminator = ""
    logger.addHandler(handler)

    # create error file handler and set level to error
    handler = logging.FileHandler(log_path, "w", encoding=None, delay="true")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    handler.terminator = ""
    logger.addHandler(handler)
