from fastapi import Request
from starlette.datastructures import MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

# machine-readable mark on a response whose text the model wrote (AI Act, article 50(2))
AI_GENERATED_HEADER = "X-AI-Generated"


def mark_ai_generated(request: Request) -> None:
    """
    Route dependency for the routes that answer with text written by the model.
    The route says so itself, so a new route can't be forgotten in a list of paths.
    """
    request.state.ai_generated = True


class AiGeneratedMiddleware:
    """
    Adds `X-AI-Generated: true` to the successful responses of the routes marked
    with `mark_ai_generated`. A raw ASGI middleware, so a streamed answer gets it
    too: the header goes out with the start of the response, after the route's
    dependencies ran.
    """

    def __init__(self, app: ASGIApp):
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_marked(message: Message) -> None:
            if (
                message["type"] == "http.response.start"
                and message["status"] < 400
                and scope.get("state", {}).get("ai_generated")
            ):
                MutableHeaders(scope=message).append(AI_GENERATED_HEADER, "true")
            await send(message)

        await self.app(scope, receive, send_marked)
