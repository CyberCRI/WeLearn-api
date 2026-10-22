import json_repair
from json_repair import JSONReturnType

from src.app.shared.utils.logger import logger as utils_logger

logger = utils_logger(__name__)


def extract_json_from_response(
    response: str,
) -> JSONReturnType | tuple[JSONReturnType, list[dict[str, str]]] | str:
    """
    Extracts JSON object from a string response.
    Args:
        response (str): The string response containing JSON.
    Returns:
        dict: The extracted JSON object.
    """
    try:
        json_data = json_repair.loads(response)
        if json_data:
            return json_data
        else:
            raise ValueError("No JSON object found in the response")
    except Exception as e:
        logger.error("api_error=json_decode_error, response=%s", response)
        raise e
