import logging
from typing import List
import transformers as tf
from globals import TORCH_BATCH_SIZE, CHUNK_FIELD, VECTOR_FIELD_BODY
from neural.BERTDot import *

log = logging.getLogger("encode")


class VectorEncoder:
    def __init__(self, pretrained_model_path):
        self.tokenizer = tf.AutoTokenizer.from_pretrained(pretrained_model_path)

        config = {
            "bert_pretrained_model": pretrained_model_path,  # 'allenai/scibert_scivocab_uncased',
            "bert_trainable": True,
        }

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print("using device:{}".format(self.device))

        self.model = BERTDot.from_config(config)

    def encode(self, text: str):
        text_encoded = self.tokenizer(
            text, return_tensors="pt", max_length=512, padding=True, truncation=True
        ).to(self.device)
        vectors = self.model.forward_representation(text_encoded)
        return vectors[0].cpu().detach().numpy()

    def encode_batch(self, texts: List[str]):
        tokens = self.tokenizer.batch_encode_plus(
            texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)

        vectors = self.model.forward_representation(tokens)
        return vectors.cpu().detach().numpy()

    def encode_documents_batch(self, documents):
        n_document = len(documents)
        print(
            "starting processing documents. number of documents to process={}".format(
                n_document
            )
        )
        for i in range(0, n_document, TORCH_BATCH_SIZE):
            batch = documents[i: i + TORCH_BATCH_SIZE]
            body_chunks = [doc[CHUNK_FIELD] for doc in batch]

            log.info(
                "encoding batch from document {} to document {}".format(
                    i, i + TORCH_BATCH_SIZE
                )
            )
            body_chunks_vectors = self.encode_batch(body_chunks)

            for j, body_chunks_vectors in enumerate(body_chunks_vectors):
                batch[j][VECTOR_FIELD_BODY] = body_chunks_vectors.tolist()
