import os

DL_WORKERS = max(2, os.cpu_count() - 4)
BATCH_SIZE = 16

DL_WORKERS = 0
