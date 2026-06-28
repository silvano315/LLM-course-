import argparse
import sys
import time

import ollama
from loguru import logger
from pydantic import BaseModel

from llm_course.data_loader.download_data import download_data
from llm_course.rag_system.constants import INSTRUCTIONS
from llm_course.vector_search.search_handler import ChunkingConfig, SearchHandler

logger.remove()


class AgentSystemResponse(BaseModel):
    answer: str
    input_tokens: int
    output_tokens: int


class AgentSystemBase:
    """
    Agent system base class that integrates a search index and an LLM client to provide answers based on tool search function.
    """
    def __init__(
        self,
        index: SearchHandler,
        llm_client: ollama.Client,
        instructions=INSTRUCTIONS,
        model='qwen3:4b'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.model = model
        self.search_tool = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the Github course database for entries matching the given query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query text to look up in the course Github database."
                        }
                    },
                    "required": ["query"]
                }
            }
        }

        logger.info(f"AgentSystemBase initialized with model: {self.model}")

    def search(self, query, num_results=5):
        boost_dict = None
        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict
        )
    
    def run(self, query):

        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": query}
        ]

        total_input = total_output = 0

        while True:
            response = self.llm_client.chat(
                model=self.model,
                messages=messages,
                tools=[self.search_tool],
                options={"num_ctx": 8192}
            )

            msg = response.message
            total_input += response.prompt_eval_count or 0
            total_output += response.eval_count or 0

            messages.append(msg)

            if not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                if tc.function.name == "search":
                    results = self.search(tc.function.arguments["query"])
                    messages.append({
                        "role": "tool",
                        "content": str(results)
                    })

        return AgentSystemResponse(
            answer=msg.content or "",
            input_tokens=total_input,
            output_tokens=total_output
        )


def main():
    parser = argparse.ArgumentParser(description="Search through the LLM Zoomcamp course materials.")
    parser.add_argument("--question", type=str, help="The question to search for Rag system.", default="How does the agentic loop keep calling the model until it stops?")
    parser.add_argument("--chunking", action="store_true", help="Whether to chunk the documents.")
    parser.add_argument("--size_chunk", type=int, default=2000, help="The size of each chunk when chunking documents.")
    parser.add_argument("--step_chunk", type=int, default=1000, help="The step size for chunking documents.")
    parser.add_argument("--level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).")

    args = parser.parse_args()

    logger.remove()
    level = args.level.upper() if isinstance(args.level, str) else str(args.level)
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )

    question = args.question

    logger.info(f"Question: {question}")

    index = SearchHandler(download_data(), chunking_config=ChunkingConfig(size_chunk=args.size_chunk, step_chunk=args.step_chunk, chunking=args.chunking))

    if args.chunking:
        logger.info(
            f"Documents chunked into {len(index.chunked_documents)} chunks of size {index.size_chunk} with step {index.step_chunk}."
        )

    agent_system = AgentSystemBase(index=index, llm_client=ollama.Client(host='http://localhost:11434', timeout=300))
    time_start = time.time()
    logger.info(f"Agent system initialized with model: {agent_system.model}")
    response = agent_system.run(question)
    logger.info(f"Answer: {response.answer}")
    logger.info(f"Input tokens: {response.input_tokens}, Output tokens: {response.output_tokens}")
    logger.info(f"Total tokens used: {response.input_tokens + response.output_tokens}")
    time_end = time.time()
    logger.info(f"Total time taken: {round(time_end - time_start, 2)} seconds")

if __name__ == "__main__":
    main()