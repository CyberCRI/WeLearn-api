import unittest

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from src.app.shared.api.ai_generated import (
    AI_GENERATED_HEADER,
    AiGeneratedMiddleware,
    mark_ai_generated,
)
from src.main import app as welearn_app

API = "/api/v1"

# the routes whose answer the model writes; a new one is added here on purpose
MARKED_ROUTES = {
    f"{API}/qna/chat/agent_stream",
    f"{API}/qna/chat/agent",
    f"{API}/tutor/files/content",
    f"{API}/tutor/mas_files/content",
    f"{API}/tutor/syllabus",
    f"{API}/tutor/syllabus/feedback",
    f"{API}/tutor/syllabus/description",
    f"{API}/tutor/syllabus/learning_objectives",
    f"{API}/tutor/syllabus/sustainability_integration",
    f"{API}/tutor/syllabus/learning_outcomes",
    f"{API}/tutor/syllabus/competency_map",
    f"{API}/tutor/api/generate",
    f"{API}/tutor/course/read",
    f"{API}/tutor/course/connection_ideas",
    f"{API}/tutor/course/connection_evidence",
    f"{API}/tutor/course/adaptation",
}


def _test_app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(AiGeneratedMiddleware)

    @app.get("/written", dependencies=[Depends(mark_ai_generated)])
    def written():
        return {"answer": "text"}

    @app.get("/streamed", dependencies=[Depends(mark_ai_generated)])
    def streamed():
        return StreamingResponse(iter(["data: a\n\n", "data: b\n\n"]))

    @app.get("/failed", dependencies=[Depends(mark_ai_generated)])
    def failed():
        raise HTTPException(status_code=429, detail="quota")

    @app.get("/plain")
    def plain():
        return {"answer": "text"}

    return app


class AiGeneratedMiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(_test_app())

    def test_marks_an_answer_the_model_wrote(self):
        response = self.client.get("/written")
        self.assertEqual(response.headers.get(AI_GENERATED_HEADER), "true")

    def test_marks_a_streamed_answer(self):
        response = self.client.get("/streamed")
        self.assertEqual(response.headers.get(AI_GENERATED_HEADER), "true")
        self.assertIn("data: b", response.text)

    def test_leaves_errors_and_other_routes_unmarked(self):
        self.assertNotIn(AI_GENERATED_HEADER, self.client.get("/failed").headers)
        self.assertNotIn(AI_GENERATED_HEADER, self.client.get("/plain").headers)


class MarkedRoutesTests(unittest.TestCase):
    def test_every_llm_route_and_only_those_are_marked(self):
        marked = {
            route.path
            for route in welearn_app.routes
            if isinstance(route, APIRoute)
            and any(
                dep.call is mark_ai_generated for dep in route.dependant.dependencies
            )
        }
        self.assertEqual(marked, MARKED_ROUTES)
