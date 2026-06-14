from gitsource import GithubRepositoryDataReader
from loguru import logger


def download_data():
    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )

    files = reader.read()

    documents = []

    for file in files:
        doc = file.parse()
        documents.append(doc)

    logger.info(f"Downloaded {len(documents)} documents")

    return documents

if __name__ == "__main__":
    docs = download_data()

    logger.info(f"Example document lesson name:\n{docs[0]['filename']}")
    logger.info(f"Example document content:\n{docs[0]['content'][:500]}...")
