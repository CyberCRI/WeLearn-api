from datetime import datetime
from enum import StrEnum, auto
from typing import Any
from uuid import UUID

from sqlalchemy import (
    TIMESTAMP,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    UniqueConstraint,
    Uuid,
    func,
    types,
)
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, relationship

Base: Any = declarative_base()


class DbSchemaEnum(StrEnum):
    CORPUS_RELATED = auto()
    DOCUMENT_RELATED = auto()
    USER_RELATED = auto()


class WeLearnDocument(Base):
    __tablename__ = "welearn_document"

    id = Column(
        Uuid(as_uuid=True),
        primary_key=True,
        server_default="gen_random_uuid()",
        nullable=False,
    )
    url = Column(String, nullable=False)
    title = Column(String)
    lang = Column(String)
    description = Column(String)
    full_content = Column(String)
    details = Column(JSON)
    trace = Column(Integer)
    corpus_id = Column(Uuid(as_uuid=True), ForeignKey("corpus_related.corpus.id"))
    created_at = Column(
        TIMESTAMP(timezone=False), nullable=False, default=func.localtimestamp()
    )
    updated_at = Column(
        TIMESTAMP(timezone=False),
        nullable=False,
        default=func.localtimestamp(),
        onupdate=func.localtimestamp(),
    )

    corpus = relationship("Corpus", foreign_keys=[corpus_id])

    __table_args__ = (
        UniqueConstraint("url", name="welearn_document_url_key"),
        {"schema": "document_related"},
    )


class Corpus(Base):
    __tablename__ = "corpus"

    id = Column(
        Uuid(as_uuid=True),
        primary_key=True,
        server_default="gen_random_uuid()",
        nullable=False,
    )
    source_name: str = Column(String, nullable=False)  # type: ignore
    is_fix: str = Column(Boolean, nullable=False)  # type: ignore
    binary_treshold: str = Column(Float, nullable=False, default=0.5)  # type: ignore

    __table_args__ = {"schema": "corpus_related"}


class CorpusEmbedding(Base):
    __tablename__ = "corpus_name_embedding_model_lang"
    source_name = Column(String, primary_key=True)
    title = Column(String)
    lang = Column(String)

    __table_args__ = {"schema": "corpus_related"}


class QtyDocumentInQdrant(Base):
    __tablename__ = "qty_document_in_qdrant"
    document_in_qdrant = Column(Integer, primary_key=True)

    __table_args__ = {"schema": "document_related"}


class DocumentSlice(Base):
    __tablename__ = "document_slice"

    id = Column(
        Uuid(as_uuid=True),
        primary_key=True,
        server_default="gen_random_uuid()",
        nullable=False,
    )
    document_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("document_related.welearn_document.id"),
        nullable=False,
    )
    embedding = Column(LargeBinary)
    body = Column(String)
    order_sequence = Column(Integer)
    embedding_model_name = Column(String)

    document = relationship("WeLearnDocument")
    __table_args__ = {"schema": "document_related"}


class Sdg(Base):
    __tablename__ = "sdg"

    id = Column(
        Uuid(as_uuid=True),
        primary_key=True,
        server_default="gen_random_uuid()",
        nullable=False,
    )
    slice_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("document_related.document_slice.id"),
        nullable=False,
    )
    sdg_number = Column(Integer, nullable=False)
    created_at = Column(
        TIMESTAMP(timezone=False), nullable=False, default=func.localtimestamp()
    )

    slice = relationship("DocumentSlice")
    __table_args__ = {"schema": "document_related"}


class UserProfile(Base):
    __tablename__ = "user_profile"
    id = Column(
        Uuid(as_uuid=True),
        primary_key=True,
        server_default="gen_random_uuid()",
        nullable=False,
    )
    username = Column(String, nullable=False)
    email = Column(String, nullable=False)
    password_digest = Column(String, nullable=False)
    created_at = Column(
        TIMESTAMP(timezone=False), nullable=False, default=func.localtimestamp()
    )
    updated_at = Column(
        TIMESTAMP(timezone=False),
        nullable=False,
        default=func.localtimestamp(),
        onupdate=func.localtimestamp(),
    )
    __table_args__ = {"schema": "user_related"}


class Bookmark(Base):
    __tablename__ = "bookmark"

    id = Column(
        Uuid(as_uuid=True),
        primary_key=True,
        server_default="gen_random_uuid()",
        nullable=False,
    )
    document_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("document_related.welearn_document.id"),
        nullable=False,
    )
    user_id = Column(
        Uuid(as_uuid=True),
        ForeignKey("user_related.user_profile.id"),
        nullable=False,
    )

    __table_args__ = {"schema": "user_related"}


class APIKeyManagement(Base):
    __tablename__ = "api_key_management"
    __table_args__ = {"schema": "user_related"}

    id: Mapped[UUID] = mapped_column(
        types.Uuid, primary_key=True, nullable=False, server_default="gen_random_uuid()"
    )
    title: Mapped[str | None]
    register_email: Mapped[str]
    digest: Mapped[bytes]
    is_active: Mapped[bool]
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False),
        nullable=False,
        default=func.localtimestamp(),
        server_default="NOW()",
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False),
        nullable=False,
        default=func.localtimestamp(),
        server_default="NOW()",
        onupdate=func.localtimestamp(),
    )


class Session(Base):
    __tablename__ = "session"
    __table_args__ = {"schema": "user_related"}
    id: Mapped[UUID] = mapped_column(
        types.Uuid, primary_key=True, nullable=False, server_default="gen_random_uuid()"
    )
    infered_user_id: Mapped[UUID] = mapped_column(
        types.Uuid,
        ForeignKey("user_related.infered_user.id"),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False),
        nullable=False,
        default=func.localtimestamp(),
        server_default="NOW()",
    )
    end_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=False), nullable=False)
    user = relationship("InferedUser", foreign_keys=[infered_user_id])


class InferredUser(Base):
    __tablename__ = "infered_user"
    __table_args__ = {"schema": "user_related"}
    id: Mapped[UUID] = mapped_column(
        types.Uuid, primary_key=True, nullable=False, server_default="gen_random_uuid()"
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False),
        nullable=False,
        default=func.localtimestamp(),
        server_default="NOW()",
    )


class EndpointRequest(Base):
    __tablename__ = "endpoint_request"
    __table_args__ = {"schema": "user_related"}
    id: Mapped[UUID] = mapped_column(
        types.Uuid, primary_key=True, nullable=False, server_default="gen_random_uuid()"
    )
    session_id: Mapped[UUID] = mapped_column(
        types.Uuid,
        ForeignKey("user_related.session.id"),
        nullable=False,
    )
    endpoint_name: Mapped[str] = mapped_column(String, nullable=False)
    http_code: Mapped[int] = mapped_column(Integer, nullable=False)
    message: Mapped[str | None] = mapped_column(String, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=False),
        nullable=False,
        default=func.localtimestamp(),
        server_default="NOW()",
    )
    session = relationship("Session", foreign_keys=[session_id])
