import argparse
import sys

from gitsource import chunk_documents
from loguru import logger
from minsearch import Index
from pydantic import BaseModel

from llm_course.data_loader.download_data import download_data

logger.remove()


class ChunkingConfig(BaseModel):
    size_chunk: int = 2000
    step_chunk: int = 1000
    chunking: bool = False


class SearchHandler:
    def __init__(self, documents, chunking_config: ChunkingConfig = ChunkingConfig()):
        self.index = Index(
            text_fields=["content"],
            keyword_fields=["filename"]
        )
        self.size_chunk = chunking_config.size_chunk
        self.step_chunk = chunking_config.step_chunk
        if not chunking_config.chunking:
            self.chunked_documents = documents
            self.index.fit(documents)
        else:
            # produce and keep the chunked documents so we can inspect/report them later
            self.chunked_documents = list(self._chunk_documents(documents))
            self.index.fit(self.chunked_documents)

    def search(self, question, boost_dict=None, filter_dict=None, num_results=5):
        return self.index.search(
            question,
            boost_dict=boost_dict,
            filter_dict=filter_dict,
            num_results=num_results
        )

    def _chunk_documents(self, documents):
        return chunk_documents(documents, size=self.size_chunk, step=self.step_chunk)


def main(question: str, num_results: int | None, chunking_config: ChunkingConfig):
    documents = download_data()
    handler = SearchHandler(documents, chunking_config=chunking_config)

    if chunking_config.chunking:
        logger.info(
            f"Documents chunked into {len(handler.chunked_documents)} chunks of size {handler.size_chunk} with step {handler.step_chunk}."
        )

    search_results = handler.search(
            question,
            boost_dict=None,
            num_results=num_results
        )
    logger.info(f"Search results for '{question}':")
    for result in search_results:
        logger.info(f" {result['filename']}: {result['content'][:100]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search through the LLM Zoomcamp course materials.")
    parser.add_argument("--question", type=str, help="The question to search for.", default="How does the agentic loop keep calling the model until it stops?")
    parser.add_argument("--num_results", type=int, default=5, help="The number of search results to return.")
    parser.add_argument("--level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")
    parser.add_argument("--chunking", action="store_true", help="Whether to chunk the documents.")
    parser.add_argument("--size_chunk", type=int, default=2000, help="The size of each chunk when chunking documents.")
    parser.add_argument("--step_chunk", type=int, default=1000, help="The step size for chunking documents.")
    args = parser.parse_args()

    logger.remove()
    level = args.level.upper() if isinstance(args.level, str) else str(args.level)
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    chunking_config = ChunkingConfig(size_chunk=args.size_chunk, step_chunk=args.step_chunk, chunking=args.chunking)
    
    main(args.question, args.num_results, chunking_config)