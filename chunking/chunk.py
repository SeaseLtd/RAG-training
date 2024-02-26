from globals import BODY_FIELD, CHUNK_FIELD, ID
import json
import copy
import spacy.cli
from langchain.text_splitter import SpacyTextSplitter
from langchain.text_splitter import CharacterTextSplitter

MAX_SPACY_SIZE = 90000

spacy.cli.download("en_core_web_sm")

char_splitter = CharacterTextSplitter(
    separator=". ",
    chunk_size=MAX_SPACY_SIZE,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False,
)

spacy_splitter = SpacyTextSplitter(chunk_size=1000)


def chunk_text(text):
    if len(text) > MAX_SPACY_SIZE:
        texts_to_chunk = char_splitter.split_text(text=text)
    else:
        texts_to_chunk = [text]

    chunked_texts = []
    for t in texts_to_chunk:
        if len(t) <= MAX_SPACY_SIZE:
            chunked_texts += spacy_splitter.split_text(text=t)

    return chunked_texts


def chunk_document(document):
    text = document[BODY_FIELD]
    chunked_documents = []
    chunked_texts = chunk_text(text)
    for i, chunk in enumerate(chunked_texts):
        tmp_document = copy.deepcopy(document)
        tmp_document[CHUNK_FIELD] = chunk
        tmp_document[ID] = str(document[ID]) + "_" + str(i)
        del tmp_document[BODY_FIELD]
        chunked_documents.append(tmp_document)
    return chunked_documents


def batch_chunk(documents):
    chunked_documents = []
    for document in documents:
        chunked_documents += chunk_document(document)

    return chunked_documents


def main():
    input_file_name = "../data/solr_documents.json"
    output_file_name = "../data/solr_documents_with_chunks.json"

    with open(input_file_name) as input_file, open(output_file_name, "w") as output_file:
        documents = json.load(input_file)
        chunked_documents = batch_chunk(documents)
        json.dump(chunked_documents, output_file)


if __name__ == "__main__":
    main()
