import logging
import os
DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
logger = logging.getLogger('READMEClassifier')

if DEBUG:
    logger.setLevel(logging.DEBUG)
    logger.warning('Running in debug mode')
else:
    logger.setLevel(logging.ERROR)