import logging
import os


_logger = logging.getLogger(__name__)
_logger.setLevel(os.environ.get("CRYSTAL_LOG_LEVEL", "warning").upper())
