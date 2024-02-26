# README #
This is the repository for all the material of the Retrieval Augmented Generation training.
Here you can find everything you need to deploy a simple RAG application based on the Solr search engine and an OpenAI LLM.

## Requirements ##
To execute the project you need:
- Docker
- An access to the OpenAI APIs

## Repository content ##
- **[chunking](chunking)**: contains the chunk.py Python script which takes in input a JSON file of Solr documents and produces a JSON file containing one Solr document for each generated chunk.
- **[data](data)**: contains input data and related Python scripts.
  - **[create.py](data/create.py)**: Python script to generate a JSON file of Solr documents.
  - **[documents_10k.tsv](data/documents_10k.tsv)**: input text for Solr documents.
  - **[solr_documents.json](data/solr_documents.json)**: JSON file of Solr documents.
  - **[solr_documents_with_chunks.json](data/solr_documents_with_chunks.json)**: JSON file of Solr chunked documents.
- **[docker-solr](docker-solr)**: contains the Docker file and the solr configuration.
- **[neural](neural)**: contains the LLM related classes.
- **[solr-tools](solr-tools)**:
  - **[index_documents.py](solr-tools/index_documents.py)**: Python script to index documents in the Solr intance.
- **[Solr.py](Solr.py)**: Solr class to instantiate a Solr instance, index documents and execute queries.
- **[SolrRetriever.py](SolrRetriever.py)**: SolrRetriever class to generate and retrieve vectors from Solr.
- **[config.yml](config.yml)**: configuration file for the uvicorn application that implements RAG.
- **[globals.py](globals.py)**: global variables.
- **[main.py](main.py)**: uvicorn application implementing the RAG pipeline.

## Configuration ##
Before proceeding, change the configuration file accordingly to your use-case.
```bash
# Solr url where the data is indexed
solr_url: http://localhost:8984/solr/rag_index

# Server port where the langchain server runs
server_port: 8000

# If you set  allenai/scibert_scivocab_uncased it is going to download it at runtime.
# If you have already downloaded the model, you can set here the local path
pretrained_model_path: allenai/scibert_scivocab_uncased

# Min size (number of characters) of a chunk to be indexed 
# All the chunks with fewer characters will be discarded
min_chunk_size: 500

# RAG endpoint
# All the fields are mandatory
endpoint:
  - 
    # OpenAI model name
    model_name: "gpt-3.5-turbo-1106"
    # The name of the endpoint. The server will be run at the address http://URL:<server_port>/<endpoint_name>/playground
    endpoint_name: "gpt_3_5_rag"
    # Number of rows to retrieve for the knn phase
    knn_rows: 10 
    # Number of rows to retrieve for the bm25 phase
    bm25_rows: 10

```

## Installation ##
To run Solr with Docker, follow the following instructions:
```bash 
cd docker-solr;
docker-compose up;
```
Solr will be available at [https://localhost:8984/](https://localhost:8984)

## Generate documents ###
**You can skip this step if you want to use the already provided material.**
To generate documents:
````
python create.py
````

To generate chunked documents:
````
python chunk.py
````

## Index documents
```bash
python index_documents.py
```
## Run RAG server
```bash
python rag.py
```

## Usage ##
TO COMPLETE
