import pickle
import numpy as np
from scipy import sparse
import gzip

print("Loading similarity matrix...")
with open('similarity.pkl', 'rb') as f:
    similarity = pickle.load(f)

print(f"Original similarity shape: {similarity.shape}")
print(f"Original size: {similarity.nbytes / 1024 / 1024:.2f} MB")

# Convert to sparse matrix to reduce size (many similarity values are close to 0)
print("Converting to sparse matrix...")
similarity_sparse = sparse.csr_matrix(similarity)

# Save compressed sparse matrix
print("Saving compressed similarity matrix...")
with gzip.open('similarity_compressed.pkl.gz', 'wb') as f:
    pickle.dump(similarity_sparse, f)

# Check compressed size
import os
compressed_size = os.path.getsize('similarity_compressed.pkl.gz')
print(f"Compressed size: {compressed_size / 1024 / 1024:.2f} MB")
print(f"Compression ratio: {compressed_size / similarity.nbytes * 100:.1f}%")

# Test loading the compressed file
print("Testing compressed file...")
with gzip.open('similarity_compressed.pkl.gz', 'rb') as f:
    similarity_loaded = pickle.load(f)
    similarity_dense = similarity_loaded.toarray()

print("✓ Compressed similarity matrix created and tested successfully!")
