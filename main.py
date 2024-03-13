from neural.VectorEncoder import VectorEncoder
import yaml
from Solr import Solr
import uvicorn
from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
)
from langserve import add_routes
from SolrRetriever import SolrRetriever


with open("./config.yml", "r") as ymlfile:
    cfg = yaml.safe_load(ymlfile)

solr = Solr(cfg["solr_url"])
endpoint = cfg["endpoint"]

vector_encoder = VectorEncoder(cfg["pretrained_model_path"])

template = """As an AI assistant specialized in question-answering tasks, your goal is to offer informative and 
accurate responses based on the provided context. Utilize passages retrieved from various sources to construct 
your answers. If the answer cannot be found within the provided documents, respond with 'Response not found.'
Be as verbose and educational in your response as possible. Each passage includes a 'source' field denoting the 
document's title it originated from. After providing an answer, be sure to cite the list of sources used for the 
corresponding passages below the answer text. Use the prefix "source:" on a new line to indicate the source of each 
cited passage.
Context: {context}
Question: "{question}"
Answer:"""
prompt = ChatPromptTemplate.from_template(template)
output_parser = StrOutputParser()

app = FastAPI(
    title="RAG Server",
    version="1.0",
    description="RAG training server",
)

openai_model = ChatOpenAI(temperature=0, model_name=endpoint[0]["model_name"])
retriever = SolrRetriever(
    solr=solr,
    vector_encoder=vector_encoder,
    retrieval_type="hybrid",
    knn_rows=endpoint[0]["knn_rows"],
    bm25_rows=endpoint[0]["bm25_rows"],
)
setup_retriever = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
)
rag_route = setup_retriever | prompt | openai_model | output_parser
add_routes(app, rag_route, path="/" + endpoint[0]["endpoint_name"])


uvicorn.run(app, host="0.0.0.0", port=cfg["server_port"])
