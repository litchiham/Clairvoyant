import os
class Config:

    # paths
    bin_path = r"D:\Project\Clairvoyant-data\bin"
    py_path = r"D:\Project\Clairvoyant-data\py"
    buffer_path = r'D:\Project\Clairvoyant-buffer'
    dust_path = r'D:\Project\Clairvoyant-dust\krigedcdod_v2'

    # log
    log_level = 'DEBUG'

    #batch
    batch_size = 64
    def __init__(self):
        pass

config=Config()