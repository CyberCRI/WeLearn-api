import logging
import os
from statistics import mean

from dotenv import load_dotenv
from ecologits.utils.range_value import RangeValue, ValueOrRange  # type: ignore

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


def format_impact_value(impact_value: ValueOrRange) -> str:
    if type(impact_value) is RangeValue:
        return str(mean([impact_value.min, impact_value.max]))
    else:
        return str(impact_value)


def log_environmental_impacts(response, logger):
    """Logs environmental impacts of LLM API calls

    Args:
        response: response from LLM API
        logger: logger object

    Returns:
        logger.info: log message
    """
    return logger.info("ECOLOGITS IS DISABLED")
    # TODO: fix ecologits error when using crew ai
    # try:
    #     impacts = {
    #         "energy": format_impact_value(response.impacts.energy.value)
    #         + " "
    #         + response.impacts.energy.unit,
    #         "gwp": format_impact_value(response.impacts.gwp.value)
    #         + " "
    #         + response.impacts.gwp.unit,
    #         "adpe": format_impact_value(response.impacts.adpe.value)
    #         + " "
    #         + response.impacts.adpe.unit,
    #         "pe": format_impact_value(response.impacts.pe.value)
    #         + " "
    #         + response.impacts.pe.unit,
    #     }
    #     return logger.info(
    #         "ecologits, energy=%s, gwp=%s, adpe=%s, pe=%s",
    #         impacts["energy"],
    #         impacts["gwp"],
    #         impacts["adpe"],
    #         impacts["pe"],
    #     )
    # except Exception as e:
    #     return logger.error("Failed to log environmental impacts: %s", e)
