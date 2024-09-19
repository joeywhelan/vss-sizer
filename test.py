import numpy as np


from redis import from_url
from redis.commands.search.field import VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

vec = np.array([2,2])
vec_bytes = vec.astype(np.float32).tobytes()
print(vec_bytes)

client = from_url('redis://@localhost:6379')
client.flushdb()

vec_params = {
            "TYPE": 'float32', 
            "DIM": 2, 
            "DISTANCE_METRIC": 'COSINE'
}

schema = [ VectorField('$.vector', 'FLAT', vec_params, as_name='vector')]
idx_def = IndexDefinition(index_type=IndexType.JSON, prefix=['key:'])
client.ft('idx').create_index(schema, definition=idx_def)
client.json().set(f'key:1', '$', {'vector': [1,1]})


qvec = np.array([1,1.1])
qvec_bytes = qvec.astype(np.float32).tobytes()
print(qvec_bytes)
q_str = f'*=>[KNN 3 @vector $query_vec AS vector_score]'
q = Query(q_str)\
        .sort_by('vector_score')\
        .paging(0,1)\
        .return_fields('vector_score')\
        .dialect(2)    
params_dict = {"query_vec": qvec_bytes}
doc = client.ft('idx').search(q, query_params=params_dict)
print(doc.docs)