from typing import Any, List
from langchain.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain.schema import BaseRetriever, Document
from globals import VECTOR_FIELD_BODY
from neural.VectorEncoder import VectorEncoder
from Solr import Solr


class SolrRetriever(BaseRetriever):
    fields: str = "bodyChunk, id"
    rows: int = 10
    vector_encoder: VectorEncoder
    solr: Solr

    def set_solr(self, solr):
        self.solr = solr

    def set_vector_encoder(self, vector_encoder):
        self.vector_encoder = vector_encoder

    def knn_query(self, query, vector_field, fields, rows):
        vector = self.vector_encoder.encode(query)
        documents = self.solr.knn_query(
            vector=vector.tolist(),
            vector_field=vector_field,
            fl=fields,
            start=0,
            rows=rows,
        )
        return documents

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> List[Document]:
        return []

    async def _aget_relevant_documents(self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun,
                                       **kwargs: Any, ) -> List[Document]:
        documents = self.knn_query(query, VECTOR_FIELD_BODY, self.fields, self.rows)
        return [
            Document(
                page_content=doc["bodyChunk"][0],
                metadata={"id": doc["id"]},
            )
            for doc in documents
        ]
