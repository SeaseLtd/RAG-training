from neural import VectorEncoder
import json
from Solr import Solr
import logging
from globals import INDEX_BATCH_SIZE, ENCODE_BATCH_SIZE
from tqdm import tqdm
import yaml

log = logging.getLogger("index_document")
logging.basicConfig()
log.setLevel(logging.INFO)


def index_document(file_name: str, solr: Solr, encoder: VectorEncoder.VectorEncoder, min_chunk_size: int):
    logging.info("Processing file: {}".format(file_name))
    input_file = open(file_name, "r")
    documents = json.load(input_file)

    documents_size = len(documents)

    for i in tqdm(range(0, documents_size, ENCODE_BATCH_SIZE)):
        tmp_documents = [d for d in documents[i: i + ENCODE_BATCH_SIZE] if len(d["bodyChunk"]) > min_chunk_size]
        encoder.encode_documents_batch(tmp_documents)
        for d in tmp_documents:
            if d.get("_version_") is not None:
                del d["_version_"]
        solr.index_documents_batch(tmp_documents, INDEX_BATCH_SIZE)

    log.info("Indexed documents in file {}".format(file_name))
    input_file.close()


def main():
    input_data = "/Users/sease/Desktop/Repositories/RAG-training/data/solr_documents_with_chunks.json"

    with open("../config.yml", "r") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    log.info("Processing file: {}".format(input_data))
    print(cfg)

    encoder = VectorEncoder.VectorEncoder(cfg["pretrained_model_path"])
    solr = Solr(cfg["solr_url"])
    min_chunk_size = cfg["min_chunk_size"]

    index_document(input_data, solr, encoder, min_chunk_size)


if __name__ == "__main__":
    main()
