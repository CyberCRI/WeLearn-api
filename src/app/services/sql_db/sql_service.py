# src/app/services/sql_db/sql_service.py

from sqlalchemy import URL, create_engine
from sqlalchemy.orm import sessionmaker

from src.app.api.dependencies import get_settings
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

        return create_engine(self.engine_url)

    def _create_session(self):

        Session = sessionmaker(bind=self.engine)
        return Session


wl_sql = WL_SQL()
session_maker = wl_sql.session_maker
