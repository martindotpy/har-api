import logging

from colorlog import ColoredFormatter

_asctime_color_format = "%(light_black)s%(asctime)s%(reset)s"
_log_level_color_format = "%(log_color)s%(levelname)s%(reset)s"
_logger_name_color_format = "%(cyan)s%(name)-20s%(reset)s"
_message_color_format = "%(message)s"

# This options could change if the app is run with no dev mode
_log_format = f"{_asctime_color_format} {_log_level_color_format} --- {_logger_name_color_format} : {_message_color_format}"
_log_profile = logging.DEBUG

_stream_handler = logging.StreamHandler()
_stream_formatter = ColoredFormatter(
    _log_format,
    log_colors={
        "DEBUG": "white",
        "INFO": "blue",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
    secondary_log_colors={
        "message": {
            "light_black": "light_black",
            "reset": "reset",
        }
    },
    style="%",
    force_color=False,
)

_stream_handler.setFormatter(_stream_formatter)

# Configure the root logger
logging.basicConfig(
    level=_log_profile,
    handlers=[_stream_handler],
)


def configure_logger(logger: logging.Logger) -> None:
    """Configure a logger with the specified format and handlers.

    Args:
        logger (logging.Logger): The logger to configure.

    """
    logger.addHandler(_stream_handler)
