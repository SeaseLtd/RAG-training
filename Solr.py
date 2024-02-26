import pysolr
from globals import DEFAULT_TOPK


class Solr:
    def __init__(self, solr_url) -> None:
        self.solr = pysolr.Solr(solr_url)

    def knn_query(self, vector, vector_field, fl, start, rows):
        top_k = DEFAULT_TOPK
        if start + rows > 50:
            top_k = start + rows
        query = "{!knn f=" + vector_field + " topK=" + str(top_k) + "}" + str(vector)
        response = self.solr.search(query, fl=fl, start=start, rows=rows)
        return response.docs

    def index_documents_batch(self, documents, batch_size):
        n_documents = len(documents)
        for i in range(0, n_documents, batch_size):
            batch = documents[i: i + batch_size]
            self.solr.add(batch, commit=True)
