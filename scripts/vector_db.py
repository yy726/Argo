import os

import duckdb
import torch

from configs.model import OUTPUT_MODEL_PATH

"""
This is a simple script to create movie embedding table in DuckDB and use the vss extension 
as the vector search engine. This is similar to the popular vector db such as Pinecone
"""

con = duckdb.connect(os.path.join(OUTPUT_MODEL_PATH, "movie_len.db"))
con.install_extension("vss")  # we need this extension to build HNSW index for ANN
con.load_extension("vss")
con.execute("SET hnsw_enable_experimental_persistence=true;")

# create the embedding table
table_creation_sql = """
    DROP TABLE IF EXISTS movie_len_embedding_table; 

    CREATE TABLE movie_len_embedding_table (
        movie_id INT,
        embedding FLOAT[64]
    );
"""
con.sql(table_creation_sql)

movie_len_embeddings = torch.load(os.path.join(OUTPUT_MODEL_PATH, "movie_embeddings_v2.pt"), weights_only=False)
batch_size = 100
num_batch = movie_len_embeddings.shape[0] // batch_size
print(f"Total number of batch {num_batch}...")
# for i in range(num_batch):
#     batch = 

# try to write a single batch
batch = movie_len_embeddings[0:100, :].tolist()
batch_idx = 0
insert_sql = "INSERT INTO movie_len_embedding_table VALUES "
values = [f"({batch_idx * batch_size + idx}, {embedding})" for idx, embedding in enumerate(batch)]
insert_sql += ",".join(values) + ";"
con.sql(insert_sql)

# create index on the table
index_creation_sql = """
    CREATE INDEX movie_len_embedding_hnsw_index
    ON movie_len_embedding_table
    USING HNSW(embedding)
    WITH (metric = 'cosine');
"""
con.sql(index_creation_sql)

# simple query check
select_sql = f"""
    SELECT *
    FROM movie_len_embedding_table
    ORDER BY array_distance(embedding, {movie_len_embeddings[0].tolist()}::FLOAT[64])
    LIMIT 5;
"""
result = con.sql(select_sql).fetchall()
print(result)

con.close()
