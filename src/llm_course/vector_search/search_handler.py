import argparse
import sys

from loguru import logger
from minsearch import Index

from llm_course.data_loader.download_data import download_data

logger.remove()
logger.add(sys.stderr, level="INFO")

class SearchHandler:
    def __init__(self, documents):
        self.index = Index(
            text_fields=["content"],
            keyword_fields=["filename"]
        )
        self.index.fit(documents)

    def search(self, question, boost_dict=None, filter_dict=None, num_results=5):
        return self.index.search(
            question,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
            num_results=num_results
        )


def main(question: str, num_results: int | None):
    documents = download_data()
    handler = SearchHandler(documents)

    search_results = handler.search(
            question,
            boost_dict={"question": 2.0, "section": 0.5},
            num_results=num_results
        )
    logger.info(f"Search results for '{question}':")
    for result in search_results:
        logger.info(f" {result['filename']}: {result['content'][:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search through the LLM Zoomcamp course materials.")
    parser.add_argument("--question", type=str, help="The question to search for.", default="How does the agentic loop keep calling the model until it stops?")
    parser.add_argument("--num_results", type=int, default=5, help="The number of search results to return.")
    
    args = parser.parse_args()
    
    main(args.question, args.num_results)
    