import logging
import os

from dotenv import load_dotenv

load_dotenv()
RUN_ENV: str | None = os.getenv("RUN_ENV", None)


class CustomFormatter(logging.Formatter):
    blue = "\x1b[38;5;75m"
    green = "\x1b[38;5;28m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    formatter = "time=(%(asctime)s) at=%(levelname)s context=%(name)s %(message)s filename=(%(filename)s:%(lineno)d)"  # noqa: E501

    FORMATS = {
        logging.DEBUG: blue + formatter + reset,
        logging.INFO: green + formatter + reset,
        logging.WARNING: yellow + formatter + reset,
        logging.ERROR: red + formatter + reset,
        logging.CRITICAL: bold_red + formatter + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def logger(name: str) -> logging.Logger:
    # Create a logger
    log = logging.getLogger(name)
    log.setLevel(logging.DEBUG)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create a formatter
    console_handler.setFormatter(CustomFormatter())

    # Add handlers to the log
    log.addHandler(console_handler)

    if RUN_ENV == "test":
        logging.disable(logging.CRITICAL)

    return log
