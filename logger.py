import logging

BASIC_FORMAT = "%(asctime)s : %(levelname)s : %(message)s"
DATA_FORMAT = '%Y-%m-%d %H:%M:%S'


class Logger(object):
    def __init__(self):
        self.formatter = logging.Formatter(BASIC_FORMAT, DATA_FORMAT)
        self.chlr = logging.StreamHandler()
        self.chlr.setFormatter(self.formatter)


    def info(self, message):
        logger = logging.getLogger('info_log')
        logger.setLevel(logging.INFO)
        logger.addHandler(self.chlr)
        logger.info(message)
    

    def debug(self, message):
        logger = logging.getLogger('debug_log')
        logger.setLevel(logging.DEBUG)
        logger.addHandler(self.chlr)
        logger.debug(message)

    def warning(self, message):
        logger = logging.getLogger('warning_log')
        logger.setLevel(logging.WARNING)
        logger.addHandler(self.chlr)
        logger.warning(message)