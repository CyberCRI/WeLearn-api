from itertools import groupby

from fastapi import BackgroundTasks
from sqlalchemy.exc import IntegrityError
from welearn_database.data.enumeration import Step

from src.app.services.sql_db.queries import (
    write_new_data_quality_error,
    write_process_state,
)
from src.app.shared.domain.documents import FullContainer
from src.app.shared.utils.decorators import log_time_and_error_sync
from src.app.shared.utils.logger import logger as logger_utils

logger = logger_utils(__name__)


class DataQualityChecker:
    def __init__(self, log_background_task):
        """
        Args:
            log_background_task: FastAPI `BackgroundTasks` used to log duplicates
                asynchronously.
        """
        self.log_background_tasks: BackgroundTasks = log_background_task

    @log_time_and_error_sync
    def remove_duplicates(
        self,
        containers_to_check: list[FullContainer],
    ) -> list[FullContainer]:
        """
        Remove duplicated containers, keeping one container per duplicate group.

        Containers are sorted then grouped by `deduplicate_hash`, so duplicates are
        detected even when they are not adjacent. For each
        group, the last container is kept and the others are logged as duplicates.
        If a background task manager is available, the logging of duplicates in
        the database is scheduled as a background task.

        Args:
            containers_to_check: Containers to deduplicate.

        Returns:
            The containers without duplicates.
        """
        ret: list[FullContainer] = []
        for k, g in groupby(
            sorted(containers_to_check, key=lambda x: x.deduplicate_hash),
            key=lambda x: x.deduplicate_hash,
        ):
            group = list(g)
            ret.append(group.pop())
            if len(group) > 0:
                logger.warning(
                    f"Duplicated {len(group)} points found in data quality check for document {k}"
                )
                if self.log_background_tasks:
                    self.log_background_tasks.add_task(
                        self._log_duplicates_containers_in_db, group
                    )
        return ret

    @staticmethod
    def _log_duplicates_containers_in_db(duplicate_containers: list[FullContainer]):
        for container in duplicate_containers:
            doc_id = container.id
            logger.info(
                f"Duplicated document {doc_id} gonna be logged in data quality errors"
            )
            try:
                ret = write_new_data_quality_error(
                    document_id=doc_id,
                    error_info="Duplicated point found in data quality check",
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
