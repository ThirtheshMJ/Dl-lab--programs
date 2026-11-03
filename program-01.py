from gensim.models import Word2Vec

# Sample document corpus
corpus = [
    ["machine", "learning", "is", "fun"],
    ["deep", "learning", "uses", "neural", "networks"],
    ["deep", "learning"]
]

# Train the Word2Vec model
model = Word2Vec(
    sentences=corpus,
    vector_size=50,
    window=2,
    min_count=1,
    workers=1,
    seed=42
)

# Print the embedding vector for a word
print("Embedding for 'learning':")
print(model.wv["learning"])

# Find similar words
print("\nWords similar to 'learning':")
print(model.wv.most_similar("learning"))