import os
import sys

# Used to create the dense document vectors.
import torch
from sentence_transformers import SentenceTransformer, models
import datasets
from tqdm import tqdm

# Used to create and store the Faiss index.
import faiss
import numpy as np
from pathlib import Path

# Dimension Reduction using PCA
from sklearn.decomposition import PCA


if len(sys.argv) == 2:
    new_dimension = int(sys.argv[1])
else:
    new_dimension = 128
model_name = f"distilbert-base-wiki-{new_dimension}"
print(f"Model name: {model_name}")

wiki_snippets = datasets.load_dataset("wiki_snippets", "wiki40b_en_100_0")
passages_length = len(wiki_snippets["train"])
device = torch.device("cuda")

if not os.path.exists(model_name):
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    if torch.cuda.is_available():
        model = model.to(device)
    print(model.device)
    wiki_snippets_shuffled = wiki_snippets.shuffle(seed=42)
    wiki_sentences = list(wiki_snippets_shuffled['train'][:1000000]["passage_text"])
    pca_train_sentences = wiki_sentences
    train_embeddings = model.encode(pca_train_sentences, convert_to_numpy=True, show_progress_bar=True)

    print("Compute the PCA")
    #Compute PCA on the train embeddings matrix
    pca = PCA(n_components=new_dimension)
    pca.fit(train_embeddings)
    pca_comp = np.asarray(pca.components_)

    dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=new_dimension, bias=False, activation_function=torch.nn.Identity())
    dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
    model.add_module('dense', dense)

    model.save(model_name)
    del(pca_train_sentences)
    del(wiki_sentences)
else:
    model = SentenceTransformer(model_name)
    if torch.cuda.is_available():
        model = model.to(device)

batch_size = 64000
steps = passages_length//batch_size
print("Start encoding the passages", passages_length, batch_size, steps)

# Convert abstracts to vectors
embeddings = None
end = 0
for step in tqdm(range(steps)):
    start, end = step*batch_size, (step+1)*batch_size
    if embeddings is None:
        embeddings = model.encode(wiki_snippets["train"][start:end]["passage_text"], show_progress_bar=True)
    else:
        embeddings = np.append(embeddings,
                           model.encode(wiki_snippets["train"][start:end]["passage_text"], show_progress_bar=True), axis=0)

if end < passages_length:
    embeddings = np.append(embeddings, 
                           model.encode(wiki_snippets["train"][end:passages_length]["passage_text"], show_progress_bar=True), axis=0)
print(embeddings.shape)

wiki_ids = np.array(range(passages_length))

# Step 1: Change data type
embeddings = np.array([embedding for embedding in embeddings]).astype("float32")

# Step 2: Instantiate the index
index = faiss.IndexFlatL2(embeddings.shape[1])

# Step 3: Pass the index to IndexIDMap
index = faiss.IndexIDMap(index)

# Step 4: Add vectors and their IDs
index.add_with_ids(embeddings, wiki_ids)

print(f"Number of vectors in the Faiss index: {index.ntotal}")

project_dir = Path('.').resolve()

index_path = f"{project_dir}/wiki_faiss_{new_dimension}.idx"
faiss.write_index(index, index_path)

id = faiss.read_index(index_path)
D, I = id.search(np.array([embeddings[50]]), k=10)
print(f'L2 distance: {D.flatten().tolist()}\n\nMAG paper IDs: {I.flatten().tolist()}')

