import numpy as np
import pandas as pd
import os
import logging

def setup(name: str = 'app_logger',log_file: str = 'app.log') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    file_handler  = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger

logger = setup()
logger.debug("debug message")
logger.info("info message")
logger.warning("warning message")
logger.error("error message")
logger.critical("critical message")


def data_load(url):
    logger.info("Data Loaded Started")
    df = pd.read_csv(url)
    logger.info("Data Load Successfully")
    return df

def save_data(data_path, df):
    logger.info("Data Save Start")                     
    save_path = os.path.join(data_path, "raw")          
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, "df_gath.csv"), index=False)
    logger.info("Data Saved Successfully")

def main():
    df = data_load('adult.csv')
    save_data("data",df)

if __name__ == "__main__":
     main()