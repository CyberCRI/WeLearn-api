import asyncio
import os
from typing import Dict, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import (
    AgentId,
    SingleThreadedAgentRuntime,
)
from autogen_core.memory import ListMemory, MemoryContent, MemoryMimeType
from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
from dotenv import load_dotenv
from pydantic import BaseModel

from src.app.services.tutor.agents import (
    UniversityTeacherAgent,
    university_teacher_topic_type,
    SDGExpertAgent,
    sdg_expert_topic_type,
    PedagogicalEngineerAgent,
    pedagogical_engineer_topic_type,
)
from src.app.services.tutor.models import Message, TutorSearchResponse


load_dotenv()


llm_4o_mini = AzureOpenAIChatCompletionClient(
    azure_deployment="gpt-4o-mini",
    model="gpt-4o-mini",
    api_version=os.environ.get("AZURE_GPT_4O_MINI_API_VERSION"),
    azure_endpoint=os.environ.get("AZURE_GPT_4O_MINI_API_BASE"),
    api_key=os.environ.get("AZURE_GPT_4O_MINI_API_KEY"),
)


async def tutor_manager(content: TutorSearchResponse) -> None:
    greencomp_memory = ListMemory()

    await greencomp_memory.add(
        MemoryContent(
            content="Here are the GreenComp competencies: "
            "1.1 Valuing sustainability: To reflect on personal values; identify and explain how values vary among people and over time, while critically evaluating how they align with sustainability values. "
            "1.2 Supporting fairness: To support equity and justice for current and future generations and learn from previous generations for sustainability. "
            "1.3 Promoting nature: To acknowledge that humans are part of nature; and to respect the needs and rights of other species and of nature itself in order to restore and regenerate healthy and resilient ecosystems. "
            "2.1 Systems thinking: To approach a sustainability problem from all sides; to consider time, space and context in order to understand how elements interact within and between systems. "
            "2.2 Critical thinking: To assess information and arguments, identify assumptions, challenge the status quo, and reflect on how personal, social and cultural backgrounds influence thinking and conclusions. "
            "2.3 Problem framing: To formulate current or potential challenges as a sustainability problem in terms of difficulty, people involved, time and geographical scope, in order to identify suitable approaches  to anticipating and preventing problems, and to mitigating and adapting to already existing problems. "
            "3.1 Futures literacy: To envision alternative sustainable futures by imagining and developing alternative scenarios and identifying the steps needed to achieve a preferred sustainable future. "
            "3.2 Adaptability: To manage transitions and challenges in complex sustainability situations and make decisions related to the future in the face of uncertainty, ambiguity and risk. "
            "3.3 Exploratory thinking: To adopt a relational way of thinking by exploring and linking different disciplines, using creativity and experimentation with novel ideas or methods. "
            "4.1 Political agency: To navigate the political system, identify political responsibility and accountability for unsustainable behaviour, and demand effective policies for sustainability. "
            "4.2 Collective action: To act for change in collaboration with others. "
            "4.3 Individual initiative: To identify own potential for sustainability and to actively contribute to improving prospects for the community and the planet.The weather should be in metric units",
            mime_type=MemoryMimeType.TEXT,
        )
    )

    runtime = SingleThreadedAgentRuntime()

    await UniversityTeacherAgent.register(
        runtime,
        type=university_teacher_topic_type,
        factory=lambda: UniversityTeacherAgent(model_client=llm_4o_mini),
    )

    await SDGExpertAgent.register(
        runtime,
        type=sdg_expert_topic_type,
        factory=lambda: SDGExpertAgent(model_client=llm_4o_mini),
    )

    await PedagogicalEngineerAgent.register(
        runtime,
        type=pedagogical_engineer_topic_type,
        factory=lambda: PedagogicalEngineerAgent(
            model_client=llm_4o_mini, memory=greencomp_memory
        ),
    )

    runtime.start()

    # response = await runtime.publish_message(
    #     content,
    #     topic_id=TopicId(university_teacher_topic_type, source="default"),
    # )
    response = await runtime.send_message(
        content,
        recipient=AgentId(university_teacher_topic_type, "default"),
    )
    print('!!!!!!!!!!!!!!!', response)

    await runtime.stop_when_idle()
    
    await runtime.close()
    return response


# asyncio.run(main())
