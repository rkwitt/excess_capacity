"""Logging"""


__all__ =  ['report_progress', 'report_status', 'report_warning', 'report_error']

import logging
from termcolor import colored

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def report_progress(s):
    logger.info(colored(s, 'green'))

def report_status(s):
    logger.info(colored(s, 'blue'))

def report_warning(s):
    logger.warning(colored(s, 'yellow'))

def report_error(s):
    logger.warning(colored(s, 'red'))

