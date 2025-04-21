import logging
import os

_logger = logging.getLogger(__name__)
_logger.setLevel(os.environ["CRYSTAL_LOG_LEVEL"].upper())
