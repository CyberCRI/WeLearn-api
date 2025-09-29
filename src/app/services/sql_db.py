from sqlalchemy import URL

from src.app.api.dependencies import get_settings
from src.app.models.db_models import EndpointRequest, MetaDocument, MetaDocumentType
from src.app.utils.decorators import singleton

settings = get_settings()


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
        from sqlalchemy.orm import sessionmaker

        Session = sessionmaker(bind=self.engine)
        return Session

    def register_endpoint(self, endpoint, session_id, http_code):
        with self.session_maker() as session:
            endpoint_request = EndpointRequest(
                endpoint_name=endpoint, session_id=session_id, http_code=http_code
            )
            session.add(endpoint_request)
            session.commit()

    def get_subject(self, subject: str) -> MetaDocument | None:
        """
        Get the subject meta document from the database.
        Args:
            subject: The subject to get.

        Returns: The subject meta document.

        """
        with self.session_maker() as session:
            subject_meta_document: MetaDocument = (
                session.query(MetaDocument)
                .join(
                    MetaDocumentType,
                    MetaDocumentType.id == MetaDocument.meta_document_type_id,
                )
                .filter(
                    MetaDocumentType.title == "subject", MetaDocument.title == subject
                )
                .first()
            )
        return subject_meta_document

    def get_subjects(self) -> list[MetaDocument]:
        """
        Get all the subject meta documents from the database.
        Returns: List of subject meta documents.
        """
        with self.session_maker() as session:
            sdg_meta_documents: list[MetaDocument] = (
                session.query(MetaDocument)
                .join(
                    MetaDocumentType,
                    MetaDocumentType.id == MetaDocument.meta_document_type_id,
                )
                .filter(MetaDocumentType.title == "subject")
                .all()
            )
        return sdg_meta_documents

    def get_meta_document(self, journey_part, sdg):
        with self.session_maker() as session:
            sdg_meta_documents: list[MetaDocument] = (
                session.query(MetaDocument)
                .join(
                    MetaDocumentType,
                    MetaDocumentType.id == MetaDocument.meta_document_type_id,
                )
                .filter(
                    MetaDocumentType.title.in_(journey_part),
                    MetaDocument.sdg_related.contains([sdg]),
                )
                .all()
            )
        return sdg_meta_documents


wl_sql = WL_SQL()
session_maker = wl_sql.session_maker
register_endpoint = wl_sql.register_endpoint
get_subject = wl_sql.get_subject
get_subjects = wl_sql.get_subjects
get_meta_document = wl_sql.get_meta_document
