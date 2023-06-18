import logging
import os
import time
import datetime


def get_logger(logger_name:str="EXP", log_folder:str="./log"):
    logger = logging.getLogger(logger_name)
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    timestamp = str(int(time.time() * 1000))
    date = datetime.datetime.now().strftime('%Y_%m_%d')
    log_date_folder = os.path.join(log_folder, date)
    os.makedirs(log_date_folder, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(log_date_folder, f"{timestamp}.txt"))
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, timestamp