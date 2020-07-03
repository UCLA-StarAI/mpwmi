
# TODO: expose API here


# LOGGING
import logging
from sys import stdout

logger = logging.getLogger("pympwmi")
stream_handler = logging.StreamHandler(stdout)
logger.addHandler(stream_handler)

def set_logger(lvl):
    logger.setLevel(lvl)

def set_logger_debug():
    set_logger(logging.DEBUG)

def set_logger_info():
    set_logger(logging.INFO)

def set_logger_file(path):
    file_handler = logging.FileHandler(path)
    logger.addHandler(file_handler)

# MAIN CLASS    
from pympwmi.mpwmi import MPWMI
from pympwmi.utils import weight_to_lit_potentials    
