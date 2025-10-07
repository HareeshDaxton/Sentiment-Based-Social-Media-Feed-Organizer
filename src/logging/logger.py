import sys
import datetime
import os
import logging

LOG_FILE = f"{datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

log_dir = os.path.join(os.getcwd(), 'logs', LOG_FILE)
os.makedirs(log_dir, exist_ok=True)

LOG_FILE_PATH = os.path.join(log_dir, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)


# import logging
# import os
# from datetime import datetime

# LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)
# os.makedirs(logs_path, exist_ok=True)

# LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# logging.basicConfig(
#     filename=LOG_FILE_PATH, 
#     format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
#     level=logging.INFO
# )
