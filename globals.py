# Performance settings
ENCODE_BATCH_SIZE = 50
INDEX_BATCH_SIZE = 10
# Batch size for transformer execution. It is important to avoid to run out of memory in case of GPU usage
TORCH_BATCH_SIZE = 2
DEFAULT_TOPK = 50

# Solr fields
BODY_FIELD = "body"
CHUNK_FIELD = "bodyChunk"
ID = "id"
VECTOR_FIELD_BODY = "bodyVector"
