import logging

logger = logging.getLogger('sox')
console = logging.StreamHandler()
console.setLevel(logging.WARNING)
logger.addHandler(console)
