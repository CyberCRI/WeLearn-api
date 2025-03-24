from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class OddishCrew:
    """Oddish crew"""

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "agents.yaml"
    tasks_config = "tasks.yaml"

    # If you would lik to add tools to your crew, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def theme_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config["theme_extractor"],
            verbose=True,
            llm="azure/gpt-4o-mini",
        )

    @agent
    def university_teacher(self) -> Agent:
        return Agent(
            config=self.agents_config["university_teacher"],
            verbose=True,
            llm="azure/gpt-4o-mini",
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def extract_input_themes(self) -> Task:
        return Task(
            config=self.tasks_config["extract_input_themes"],
        )

    @task
    def generate_course_plan(self) -> Task:
        return Task(
            config=self.tasks_config["generate_course_plan"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Oddish Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            memory=False,
        )
