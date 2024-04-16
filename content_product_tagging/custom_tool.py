from typing import Optional
from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from util import prompt_search

class ImageSearchResults(BaseTool):
    """Tool that queries the Fashion Image Search API and gets back json."""

    name: str = "image_search_results_json"
    description: str = (
        "A wrapper around Image Search. "
        "Useful for when you need search fashion images related to cloth , shoe etc"
        "Input should be a search query. Output is a JSON array of the query results"
    )
    num_results: int = 4

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return str(prompt_search(prompt = query, topn=self.num_results))