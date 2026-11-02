from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_assistant.pipeline import CHUNK_OVERLAP, CHUNK_SIZE


def test_splitter_uses_resume_chunk_size_and_overlap():
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    parts = splitter.split_text("A" * 1600)
    assert max(map(len, parts)) <= 800
    assert parts[0][-150:] == parts[1][:150]
