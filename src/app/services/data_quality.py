import hashlib

from qdrant_client.conversions.common_types import ScoredPoint

from src.app.utils.decorators import log_time_and_error_sync
from src.app.utils.logger import logger as logger_utils

logger = logger_utils(__name__)


class DataQualityChecker:
    def __init__(self):
        pass


@log_time_and_error_sync
def remove_duplicates(
    keys_to_check: list[str], points_to_check: list[ScoredPoint], strict: bool = False
) -> list[ScoredPoint]:
    """
    Remove duplicated points according to the keys in the payload. If strict is True, raise an error when a key is missing or not a string.
    Args:
        keys_to_check: list of keys to check in the payload /!\ Must be first level keys, not keys into another dict
        points_to_check: list of points to check
        strict: whether to raise an error when a key is missing or not a string
    Returns: list of points without duplicates
    """
    ret: list[ScoredPoint] = []
    hashes = set()
    for point in points_to_check:
        payload = point.payload

        if not payload:
            logger.error(f"Point {point.id} doesn't have payload")
            continue

        values_to_check = []
        for key in keys_to_check:
            if not key in payload:
                logger.error(f"Point {point.id} doesn't have key {key}")
                if strict:
                    raise ValueError(f"Point {point.id} doesn't have key {key}")
                continue
            if not isinstance(payload[key], str):
                msg = f"Data quality deduplication can be only applied on string, {key} is {type(payload[key])}"
                if strict:
                    raise TypeError(msg)
                logger.error(msg + f" key {key} will be ignored")
            else:
                values_to_check.append(str(payload[key]).encode("utf-8"))
        local_hash = hashlib.sha256()
        for vtc in sorted(values_to_check):
            local_hash.update(vtc)

        hexdigest = local_hash.hexdigest()
        if hexdigest in hashes:
            logger.error(
                f"Point {point.id} is duplicated, according these keys : {keys_to_check}"
            )
        else:
            hashes.add(hexdigest)
            ret.append(point)
    return ret
