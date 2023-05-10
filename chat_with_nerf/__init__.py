import logging.config


class ColorfulTextFormatter(logging.Formatter):
    """Custom colorful text formatter for CLI."""

    def __init__(self, my_format) -> None:
        super().__init__()
        grey = "\x1b[38;20m"
        green = "\x1b[32;20m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"
        self.my_format = my_format

        self.FORMATS = {
            logging.DEBUG: grey + self.my_format + reset,
            logging.INFO: green + self.my_format + reset,
            logging.WARNING: yellow + self.my_format + reset,
            logging.ERROR: red + self.my_format + reset,
            logging.CRITICAL: bold_red + self.my_format + reset,
        }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logging_dict_config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "json": {
            "format": '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": " %(name)s", "func_name": "%(funcName)s", "filename":  "%(filename)s:%(lineno)s", "message": "%(message)s"}'  # noqa: E501
        },
        "plaintext": {
            "format": "[%(asctime)s] %(levelname)s %(name)s [%(funcName)s] [%(filename)s:%(lineno)s] - %(message)s"  # noqa: E501
        },
        "colorfultext": {
            "()": ColorfulTextFormatter,
            "my_format": "[%(asctime)s] %(levelname)s %(name)s [%(funcName)s] [%(filename)s:%(lineno)s] - %(message)s",  # noqa: E501
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "colorfultext",
            "stream": "ext://sys.stdout",
        },
        "logfile": {
            "class": "logging.handlers.TimedRotatingFileHandler",
            "formatter": "plaintext",
            "filename": "chat2ground.log",
            "when": "D",
            "backupCount": 3,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "logfile"],
    },
    "loggers": {
        "nerf_grounding_chat_interface": {"level": "DEBUG"},
    },
}

logging.config.dictConfig(logging_dict_config)
logger = logging.getLogger(__name__)
