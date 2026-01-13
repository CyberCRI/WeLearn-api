# src/app/services/sql_service.py

import uuid
from threading import Lock
from uuid import UUID

from sqlalchemy import URL
from sqlalchemy.orm import sessionmaker
from welearn_database.data.enumeration import Step
from welearn_database.data.models import (
    ContextDocument,
    EmbeddingModel,
    EndpointRequest,
    ErrorDataQuality,
    ProcessState,
)

from src.app.api.dependencies import get_settings
from src.app.models.documents import JourneySection
from src.app.models.search import ContextType
from src.app.services.constants import APP_NAME
from src.app.utils.decorators import singleton

# src/app/services/sql_service.py


settings = get_settings()

model_id_cache: dict[str, UUID] = {}
model_id_lock = Lock()


@singleton
class WL_SQL:
    def __init__(self):
        self.engine_url = URL.create(
            drivername=settings.PG_DRIVER,
            username=settings.PG_USER or None,
            password=settings.PG_PASSWORD or None,
            host=settings.PG_HOST or None,
            port=int(settings.PG_PORT) if settings.PG_PORT else None,
            database=settings.PG_DATABASE,
        )
        self.engine = self._create_engine()
        self.session_maker = self._create_session()

    def _create_engine(self):
        from sqlalchemy import create_engine

        return create_engine(self.engine_url)

    def _create_session(self):

        Session = sessionmaker(bind=self.engine)
        return Session

    def register_endpoint(self, endpoint, session_id, http_code):
        with self.session_maker() as session:
            endpoint_request = EndpointRequest(
                endpoint_name=endpoint, session_id=session_id, http_code=http_code
            )
            session.add(endpoint_request)
            session.commit()

    def get_subject(
        self, subject: str, embedding_model_id: UUID
    ) -> ContextDocument | None:
        """
        Get the subject meta document from the database.
        Args:
            subject: The subject to get.

        Returns: The subject meta document.

        """
        with self.session_maker() as session:
            subject_meta_document: ContextDocument | None = (
                session.query(ContextDocument)
                .filter(
                    ContextDocument.context_type == ContextType.SUBJECT.value.lower(),
                    ContextDocument.title == subject,
                    ContextDocument.embedding_model_id == embedding_model_id,
                )
                .first()
            )
        return subject_meta_document

    def get_subjects(self, embedding_model_id: UUID) -> list[ContextDocument]:
        """
        Get all the subject meta documents from the database.
        Returns: List of subject meta documents.
        """
        with self.session_maker() as session:
            sdg_meta_documents: list[ContextDocument] = (
                session.query(ContextDocument)
                .filter(
                    ContextDocument.context_type == ContextType.SUBJECT.value.lower(),
                    ContextDocument.embedding_model_id == embedding_model_id,
                )
                .all()
            )
        return sdg_meta_documents

    def get_context_documents(
        self, journey_part: JourneySection, sdg: int, embedding_model_id: UUID
    ):
        """
        Get the context documents from the database.

        Args:
            journey_part: The journey part to get the context documents for.
            sdg: The SDG to get the context documents for.
        Returns: List of context documents.
        """
        with self.session_maker() as session:
            sdg_meta_documents: list[ContextDocument] = (
                session.query(ContextDocument)
                .filter(
                    ContextDocument.context_type.in_(journey_part),
                    ContextDocument.sdg_related.contains([sdg]),
                    ContextDocument.embedding_model_id == embedding_model_id,
                )
                .all()
            )
        return sdg_meta_documents

    def get_embeddings_model_id_according_name(self, model_name: str) -> UUID | None:
        """
        Get the embeddings model ID according to its name.

        Args:
            model_name: The name of the embeddings model.

        Returns:
            The ID of the embeddings model if found, otherwise None.
        """
        with model_id_lock:
            if model_name in model_id_cache:
                return model_id_cache[model_name]

        with self.session_maker() as session:
            model = (
                session.query(EmbeddingModel)
                .filter(EmbeddingModel.title == model_name)
                .first()
            )
            return model.id if model else None

    def write_new_data_quality_error(
        self, document_id: UUID, error_info: str, slice_id: UUID | None = None
    ) -> UUID:
        """
        Write a new data quality error to the database.
        Args:
            document_id: The ID of the document with the error.
            slice_id: The ID of the document slice with the error.
            error_info:  The error information. Usually exception message.

        Returns:
            The ID of the new error entry.
        """
        with self.session_maker() as session:
            error_entry = ErrorDataQuality(
                id=uuid.uuid4(),
                document_id=document_id,
                slice_id=slice_id,
                error_raiser=APP_NAME,
                error_info=error_info,
            )
            session.add(error_entry)
            session.commit()
            return error_entry.id

    def write_process_state(self, document_id: UUID, process_state: Step) -> UUID:
        """
        Write the process state of a document to the database.
        Args:
            document_id: The ID of the document.
            process_state: The current process state.

        Returns:
            The ID of the new process state entry.
        """
        with self.session_maker() as session:
            process_state_entry = ProcessState(
                id=uuid.uuid4(),
                document_id=document_id,
                title=process_state.value.lower(),
            )
            session.add(process_state_entry)
            session.commit()
            return process_state_entry.id


wl_sql = WL_SQL()
session_maker = wl_sql.session_maker
register_endpoint = wl_sql.register_endpoint
get_subject = wl_sql.get_subject
get_subjects = wl_sql.get_subjects
get_context_documents = wl_sql.get_context_documents
get_embeddings_model_id_according_name = wl_sql.get_embeddings_model_id_according_name
write_new_data_quality_error = wl_sql.write_new_data_quality_error
write_process_state = wl_sql.write_process_state
