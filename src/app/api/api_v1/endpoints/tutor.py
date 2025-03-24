from typing import Annotated, Optional

from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel

from src.app.api.dependencies import get_settings
from src.app.services.abst_chat import AbstractChat, ChatFactory
from src.app.services.agents import (
    theme_extractor,
    theme_extractor_action,
    university_teacher,
    university_teacher_action,
)
from src.app.services.helpers import stringify_docs_content
from src.app.services.search import SearchService
from src.app.services.search_helpers import search_multi_inputs
from src.app.utils.logger import logger as utils_logger
from src.crews.crew_manager import kickoff

logger = utils_logger(__name__)

router = APIRouter()

settings = get_settings()

chatfactory: AbstractChat = ChatFactory().create_chat("openai")
chatfactory.init_client()

sp = SearchService()


class OddishParams(BaseModel):
    file: Optional[UploadFile] = File(default=None)


@router.post("/oddish_crew")
async def kickoff_crew_endpoint(files: Annotated[list[UploadFile], File()]):
    # get sources and use chat to classify text in parallel
    file_content: list[bytes] = [await file.read() for file in files]
    file_content_str = [content.decode("utf-8") for content in file_content]
    docs = await search_multi_inputs(
        inputs=file_content_str, nb_results=5, callback_function=sp.search
    )

    result = await kickoff(text_contents=file_content_str, search_results=docs)

    return result


@router.post("/oddish")
async def oddish(files: Annotated[list[UploadFile], File()]):
    print(files)
    file_content: list[bytes] = [await file.read() for file in files]
    file_content_str = [content.decode("utf-8") for content in file_content]
    docs = await search_multi_inputs(
        inputs=file_content_str, nb_results=5, callback_function=sp.search
    )
    text_contents = "//n".join(file_content_str)

    theme = await chatfactory.flex_chat(
        messages=[
            {"role": "system", "content": theme_extractor},
            {
                "role": "user",
                "content": theme_extractor_action.format(text_contents=text_contents),
            },
        ]
    )

    stringified_docs = stringify_docs_content(docs or [])

    plan = await chatfactory.flex_chat(
        messages=[
            {"role": "system", "content": university_teacher},
            {
                "role": "user",
                "content": university_teacher_action.format(
                    search_results=stringified_docs,
                    text_contents=text_contents,
                    theme=theme,
                ),
            },
        ]
    )

    return plan
