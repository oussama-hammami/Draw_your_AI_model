"""Logging configuration for the application."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure structured logging for the application.

    Args:
        level: The logging level to use. Defaults to INFO.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(handler)
