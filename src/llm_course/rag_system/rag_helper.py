import argparse

import ollama
from loguru import logger
from pydantic import BaseModel

from llm_course.data_loader.download_data import download_data
from llm_course.rag_system.constants import INSTRUCTIONS, PROMPT_TEMPLATE
from llm_course.vector_search.search_handler import SearchHandler


class RAGResponse(BaseModel):
    answer: str
    input_tokens: int
    output_tokens: int


class RAGBase:

    def __init__(
        self,
        index: SearchHandler,
        llm_client: ollama.Client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        model='deepseek-r1:1.5b'
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

        logger.info(f"RAGBase initialized with model: {self.model}")

    def search(self, query, num_results=5):
        boost_dict = None
        return self.index.search(
            query,
            num_results=num_results,
            boost_dict=boost_dict
        )

    def build_context(self, search_results):
        lines = []

        for i, doc in enumerate(search_results):
            lines.append('File name: ' + doc['filename'])
            lines.append('Content: ' + doc['content'])
            lines.append('')
            logger.info(f"Context built for file {i + 1}: {doc['filename']}")
            logger.info(f"Content snippet {i + 1}: {doc['content'][:100]}...")

        return '\n'.join(lines).strip()

    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):

        response = self.llm_client.generate(
            model=self.model,
            prompt=prompt,
            system=self.instructions,
            options={"num_ctx": 8192}
        )

        return response.get('response'), response.prompt_eval_count, response.eval_count

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer, input_tokens, output_tokens = self.llm(prompt)
        return RAGResponse(answer=answer, input_tokens=input_tokens, output_tokens=output_tokens)

def main():
    parser = argparse.ArgumentParser(description="Search through the LLM Zoomcamp course materials.")
    parser.add_argument("--question", type=str, help="The question to search for Rag system.", default="How does the agentic loop keep calling the model until it stops?")

    args = parser.parse_args()

    question = args.question

    logger.info(f"Question: {question}")

    index = SearchHandler(download_data())

    rag_system = RAGBase(index=index, llm_client=ollama.Client(host='http://localhost:11434', timeout=300))
    response = rag_system.rag(question)
    logger.info(f"Answer: {response.answer}")
    logger.info(f"Input tokens: {response.input_tokens}, Output tokens: {response.output_tokens}")

if __name__ == "__main__":
    main()