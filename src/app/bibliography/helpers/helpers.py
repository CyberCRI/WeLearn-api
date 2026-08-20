import datetime
from typing import Any, Optional


from welearn_database.data.models import WeLearnDocument

from src.app.utils.decorators import log_time_and_error_sync
from src.app.utils.logger import logger as utils_logger

logger = utils_logger(__name__)


def ris_line(tag: str, value: Any) -> Optional[str]:
    """Formate une ligne RIS, ou None si la valeur est vide."""
    if value is None or value == "":
        return None
    return f"{tag}  - {value}"


def compute_ris_doctype(doc_type: Any | None) -> str:
    match doc_type:
        case "book":
            return "BOOK"
        case "chapter":
            return "CHAP"
        case "article":
            return "JFULL"
        case _:
            return "ELEC"


def compute_publication_date_for_ris(pub_date: str | int | float) -> str:
    try:
        tmp_ret = datetime.date.fromtimestamp(int(pub_date)).isoformat()
        return tmp_ret.replace("-", "/")
    except Exception as e:
        logger.warning(
            "Exception occurs during publication formatting, field returned empty: %s",
            e,
        )
        return ""


def compute_authors_for_ris(authors: list[dict[str, Any]]) -> list[str]:
    ret: list[str] = []
    for author in authors:
        line = ris_line("AU", author.get("name"))
        if line:
            ret.append(line)
    return ret


@log_time_and_error_sync
def welearn_document_to_ris(doc: WeLearnDocument) -> str:
    details = doc.details or {}

    lines: list[str] = []

    doc_type = details.get("type", None)
    lines.append(ris_line("TY", compute_ris_doctype(doc_type)))
    lines.append(ris_line("ID", doc.id))
    lines.append(ris_line("TI", doc.title))
    lines.append(ris_line("UR", doc.url))
    lines.append(ris_line("AB", doc.description))
    lines.append(ris_line("DB", doc.corpus.source_name))
    lines.append(ris_line("LA", doc.lang))
    pub_date = details.get("publication_date", None)
    if pub_date:
        ris_pub_date = compute_publication_date_for_ris(pub_date=pub_date)
        if ris_pub_date:
            lines.append(ris_line("PY", ris_pub_date))
    if doc.doi:
        lines.append(ris_line("DO", doc.doi))
    authors = details.get("authors", None)
    if authors:
        lines.extend(compute_authors_for_ris(authors=authors))
    publisher = details.get("publisher", None)
    if publisher:
        lines.append(ris_line("PB", publisher))
    license_url = details.get("license_url", None)
    if license_url:
        lines.append(ris_line("C1", license_url))
    lines.append("ER  - ")
    return "\n".join([line for line in lines if line])
