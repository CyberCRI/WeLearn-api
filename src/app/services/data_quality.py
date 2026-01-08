import hashlib
from typing import Any

from _hashlib import HASH
from fastapi import BackgroundTasks
from qdrant_client.http.models import ScoredPoint
from sqlalchemy.exc import IntegrityError
from welearn_database.data.enumeration import Step

from src.app.services.sql_service import (
    write_new_data_quality_error,
    write_process_state,
)

# from src.app.services.sql_db import write_new_data_quality_error, write_process_state
from src.app.utils.decorators import log_time_and_error_sync
from src.app.utils.logger import logger as logger_utils

logger = logger_utils(__name__)


class DataQualityChecker:
    def __init__(self, log_background_task):
        self.log_background_tasks: BackgroundTasks = log_background_task

    @log_time_and_error_sync
    def remove_duplicates(
        self,
        keys_to_check: list[str],
        points_to_check: list[ScoredPoint],
        strict: bool = False,
    ) -> list[ScoredPoint]:
        """
        Remove duplicated points according to the keys in the payload. If strict is True, raise an error when a key is missing or not a string.
        Args:
            keys_to_check: list of keys to check in the payload :warning: Must be first level keys, not keys into another dict
            points_to_check: list of points to check
            strict: whether to raise an error when a key is missing or  not a string
        Returns: list of points without duplicates
        """
        if len(keys_to_check) <= 0:
            msg = (
                "The method need a key to check duplicates and 'keys_to_check' is empty"
            )
            logger.error(msg)
            if strict:
                raise ValueError(msg)
            logger.info("Method return points without de-duplication")
            return points_to_check

        ret: list[ScoredPoint] = []
        hashes = set()
        for point in points_to_check:
            payload = point.payload

            if not payload:
                logger.error(
                    f"Point {point.id} doesn't have payload, deduplication ignored"
                )
                ret.append(point)
                continue

            local_hash = self.compute_hashes(keys_to_check, payload, point, strict)

            hexdigest = local_hash.hexdigest()
            if hexdigest in hashes:
                logger.info(
                    f"Point {point.id} is duplicated, according these keys : {keys_to_check}"
                )
            else:
                hashes.add(hexdigest)
                ret.append(point)

        if self.log_background_tasks:
            self.log_background_tasks.add_task(
                self._log_duplicates_points_in_db, points_to_check, ret
            )
        return ret

    def compute_hashes(
        self,
        keys_to_check: list[str],
        payload: dict[str, Any],
        point: ScoredPoint,
        strict: bool,
    ) -> HASH:
        local_hash = hashlib.sha256()
        values_to_check = self.retrieve_values_to_check(
            keys_to_check, payload, point, strict
        )
        for vtc in sorted(values_to_check):
            local_hash.update(vtc)
        return local_hash

    def retrieve_values_to_check(
        self,
        keys_to_check: list[str],
        payload: dict[str, Any],
        point: ScoredPoint,
        strict: bool,
    ):
        values_to_check: list = []
        for key in keys_to_check:
            if key not in payload:
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
                values_to_check.append(
                    str(key).encode("utf-8") + str(payload[key]).encode("utf-8")
                )
        return values_to_check

    @staticmethod
    def _log_duplicates_points_in_db(
        original_list: list[ScoredPoint], deduplicated_list: list[ScoredPoint]
    ):
        duplicated_points = [p for p in original_list if p not in deduplicated_list]
        documents_ids = set()
        for dp in duplicated_points:
            document_id = dp.payload.get("document_id", None)
            if not document_id:
                logger.error(
                    f"Duplicated point found: {dp.id} but no document id in payload"
                )
                continue
            documents_ids.add(document_id)
        logger.info(f"Total duplicated points found: {len(duplicated_points)}")

        for doc_id in documents_ids:
            logger.info(
                f"Duplicated document {doc_id} gonna be logged in data quality errors"
            )
            try:
                ret = write_new_data_quality_error(
                    document_id=doc_id,
                    error_info=f"Duplicated point found in data quality check: {dp.id}",
                )
                logger.info(f"Data quality error logged with id: {ret}")
            except IntegrityError as e:
                logger.info(
                    f"Point {doc_id} already logged in data quality errors: {str(e.orig)}"
                )
                continue

            logger.info(f"Document id with duplicated points: {doc_id}")
            ret = write_process_state(
                document_id=doc_id, process_state=Step.DOCUMENT_IS_INVALID
            )
            logger.info(f"Process state updated with id: {ret}")
