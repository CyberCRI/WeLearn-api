import os

from crewai.flow import Flow, start
from pydantic import BaseModel

from src.crews.oddish_crew import OddishCrew

# Optional: Use environment variables for API keys
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"


class OddishState(BaseModel):
    # Note: 'id' field is automatically added to all states
    text_contents: list[str] = []
    resources: list = []
    course: str = ""


class OddishFlow(Flow[OddishState]):
    @start()
    def draft_course(self):
        print("Drafting course")
        course = (
            OddishCrew()
            .crew()
            .kickoff(
                inputs={
                    "text_contents": self.state.text_contents,
                    "search_results": self.state.resources,
                }
            )
        )
        self.state.course = course.raw
        return self.state.course


async def kickoff(text_contents, search_results):
    oddish_flow = OddishFlow()
    return await oddish_flow.kickoff_async(
        inputs={"text_contents": text_contents, "search_results": search_results}
    )


def plot():
    oddish_flow = OddishFlow()
    oddish_flow.plot()
