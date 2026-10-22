# src/app/shared/infra/queries.py
from welearn_database.data.models import EmbeddingModel, EndpointRequest

from src.app.shared.infra.db import session_maker


def register_endpoint(endpoint, session_id, http_code):
    with session_maker() as session:
        endpoint_request = EndpointRequest(
            endpoint_name=endpoint, session_id=session_id, http_code=http_code
        )
        session.add(endpoint_request)
        session.commit()


def get_embeddings_model_id_according_name(
    model_name: str,
) -> list[EmbeddingModel | None]:
    """
    Get the embeddings model ID according to its name.

    Args:
        model_name: The name of the embeddings model.

    Returns:
        The ID of the embeddings model if found, otherwise None.
    """
    with session_maker() as session:
        return (
            session.query(EmbeddingModel)
            .filter(EmbeddingModel.title == model_name)
            .all()
        )
