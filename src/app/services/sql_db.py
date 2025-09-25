from sqlalchemy import URL

from src.app.api.dependencies import get_settings
from src.app.models.db_models import EndpointRequest
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


wl_sql = WL_SQL()
session_maker = wl_sql.session_maker
register_endpoint = wl_sql.register_endpoint
