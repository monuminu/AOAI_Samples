from typing import Optional, List, Dict, Any, Set

from gpt_researcher.config import Config
from gpt_researcher.memory import Memory
from gpt_researcher.utils.enum import ReportSource, ReportType, Tone
from gpt_researcher.llm_provider import GenericLLMProvider
from gpt_researcher.master.agent.researcher import ResearchConductor
from gpt_researcher.master.agent.scraper import ReportScraper
from gpt_researcher.master.agent.writer import ReportGenerator
from gpt_researcher.master.agent.context_manager import ContextManager
from gpt_researcher.master.actions import get_retrievers, choose_agent
from gpt_researcher.vector_store import VectorStoreWrapper


class GPTResearcher:
    def __init__(
        self,
        query: str,
        report_type: str = ReportType.ResearchReport.value,
        report_format: str = "markdown",  # Add this line
        report_source: str = ReportSource.Web.value,
        tone: Tone = Tone.Objective,
        source_urls=None,
        documents=None,
        vector_store=None,
        vector_store_filter=None,
        config_path=None,
        websocket=None,
        agent=None,
        role=None,
        parent_query: str = "",
        subtopics: list = [],
        visited_urls: set = set(),
        verbose: bool = True,
        context=[],
        headers: dict = None,
        max_subtopics: int = 5,  # Add this line
    ):
        self.query = query
        self.report_type = report_type
        self.cfg = Config(config_path)
        self.llm = GenericLLMProvider(self.cfg)
        self.report_source = getattr(
            self.cfg, 'report_source', None) or report_source
        self.report_format = report_format
        self.max_subtopics = max_subtopics
        self.tone = tone if isinstance(tone, Tone) else Tone.Objective
        self.source_urls = source_urls
        self.documents = documents
        self.vector_store = VectorStoreWrapper(vector_store) if vector_store else None
        self.vector_store_filter = vector_store_filter
        self.websocket = websocket
        self.agent = agent
        self.role = role
        self.parent_query = parent_query
        self.subtopics = subtopics
        self.visited_urls = visited_urls
        self.verbose = verbose
        self.context = context
        self.headers = headers or {}
        self.research_costs = 0.0
        self.retrievers = get_retrievers(self.headers, self.cfg)
        self.memory = Memory(
            getattr(self.cfg, 'embedding_provider', None), self.headers)

        # Initialize components
        self.research_conductor = ResearchConductor(self)
        self.report_generator = ReportGenerator(self)
        self.scraper = ReportScraper(self)
        self.context_manager = ContextManager(self)

    async def conduct_research(self):
        if not (self.agent and self.role):
            self.agent, self.role = await choose_agent(
                query=self.query,
                cfg=self.cfg,
                parent_query=self.parent_query,
                cost_callback=self.add_costs,
                headers=self.headers,
            )

        self.context = await self.research_conductor.conduct_research()
        return self.context

    async def write_report(self, existing_headers: list = [], relevant_written_contents: list = [], ext_context=None) -> str:
        return await self.report_generator.write_report(
            existing_headers,
            relevant_written_contents,
            ext_context or self.context
        )

    async def write_report_conclusion(self, report_body: str) -> str:
        return await self.report_generator.write_report_conclusion(report_body)

    async def write_introduction(self):
        return await self.report_generator.write_introduction()

    async def get_subtopics(self):
        return await self.report_generator.get_subtopics()

    async def get_draft_section_titles(self, current_subtopic: str):
        return await self.report_generator.get_draft_section_titles(current_subtopic)

    async def get_similar_written_contents_by_draft_section_titles(
        self,
        current_subtopic: str,
        draft_section_titles: List[str],
        written_contents: List[Dict],
        max_results: int = 10
    ) -> List[str]:
        return await self.context_manager.get_similar_written_contents_by_draft_section_titles(
            current_subtopic,
            draft_section_titles,
            written_contents,
            max_results
        )

    # Utility methods
    def get_source_urls(self) -> list:
        return list(self.visited_urls)

    def get_research_context(self) -> list:
        return self.context

    def get_costs(self) -> float:
        return self.research_costs

    def set_verbose(self, verbose: bool):
        self.verbose = verbose

    def add_costs(self, cost: float) -> None:
        if not isinstance(cost, (float, int)):
            raise ValueError("Cost must be an integer or float")
        self.research_costs += cost
