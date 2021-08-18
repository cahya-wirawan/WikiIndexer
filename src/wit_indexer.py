import random
import os
import sys
import pickle
from random import randint
from wit_dataset import WitDataset

# Used to create the dense document vectors.
import torch
from sentence_transformers import SentenceTransformer, models
from tqdm import tqdm

# Used to create and store the Faiss index.
import faiss
import numpy as np
from pathlib import Path

# Dimension Reduction using PCA
from sklearn.decomposition import PCA

WIT_dir = "/mnt/mldata/data/WIT/test"
# WIT_files = ["wit_v1.train.all-1percent_sample.tsv"]
# WIT_files = ["test.tsv"]
WIT_files = ["wit_v1.train.all-00000-of-00010.tsv"]


def create_sentence_model(model_name, pct=1.0):
    if not os.path.exists(model_name):
        model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

        if torch.cuda.is_available():
            model = model.to(device)
        print(model.device)
        # wiki_snippets_shuffled = wiki_snippets.shuffle(seed=42)
        wiki_sentences = descriptions[:int(pct * len(descriptions))]
        pca_train_sentences = wiki_sentences
        train_embeddings = model.encode(pca_train_sentences, convert_to_numpy=True, show_progress_bar=True)

        print("Compute the PCA")
        # Compute PCA on the train embeddings matrix
        pca = PCA(n_components=new_dimension)
        pca.fit(train_embeddings)
        pca_comp = np.asarray(pca.components_)

        dense = models.Dense(in_features=model.get_sentence_embedding_dimension(), out_features=new_dimension,
                             bias=False, activation_function=torch.nn.Identity())
        dense.linear.weight = torch.nn.Parameter(torch.tensor(pca_comp))
        model.add_module('dense', dense)

        model.save(model_name)
        del pca_train_sentences
        del wiki_sentences
    else:
        model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            model = model.to(device)

    return model


wit_dataset = {
    "image_info": {},
    "desc2image_map": []
}
if len(sys.argv) == 2:
    new_dimension = int(sys.argv[1])
else:
    new_dimension = 128
model_name = f"distilbert-base-wiki-{new_dimension}"
print(f"Model name: {model_name}")

device = torch.device("cuda")
wit_counter = 0
model = None
batch_size = 64000
wiki_ids = []

# Instantiate the index
index = faiss.IndexFlatL2(new_dimension)

# Pass the index to IndexIDMap
index = faiss.IndexIDMap(index)
descriptions_test = []
max_descriptions_test_length = 10
last_dataframe_size = 0

for wit_file in sorted(Path(WIT_dir).glob("*.tsv")):
    print(wit_file)
    descriptions, image_info, desc2image_map, dataframe_size = WitDataset.read(wit_file)
    image_info = {key+last_dataframe_size: image_info[key] for key in image_info}
    wit_dataset["image_info"] = {**wit_dataset["image_info"], **image_info}
    desc2image_map = [i+last_dataframe_size for i in desc2image_map]
    wit_dataset["desc2image_map"] = wit_dataset["desc2image_map"] + desc2image_map
    last_dataframe_size += dataframe_size
    passages_length = len(descriptions)
    if wit_counter == 0:
        model = create_sentence_model(model_name, pct=0.25)

    while passages_length <= batch_size:
        batch_size = batch_size >> 1
    print(f"\nBatch size: {batch_size}")
    steps = passages_length // batch_size
    print("Start encoding the passages", passages_length, batch_size, steps)
    # Convert abstracts to vectors
    embeddings = None
    end = 0
    for step in tqdm(range(steps)):
        start, end = step * batch_size, (step + 1) * batch_size
        if embeddings is None:
            texts = descriptions[start:end]
            embeddings = model.encode(texts, show_progress_bar=True)
        else:
            texts = descriptions[start:end]
            embeddings = np.append(embeddings,
                                   model.encode(texts, show_progress_bar=True), axis=0)

    if end < passages_length:
        embeddings = np.append(embeddings,
                               model.encode(descriptions[end:passages_length], show_progress_bar=True), axis=0)
    print(embeddings.shape)
    wiki_ids = np.array(range(index.ntotal, index.ntotal+passages_length))

    # Add vectors and their IDs
    index.add_with_ids(embeddings, wiki_ids)
    wit_counter += 1

project_dir = Path('.').resolve()
pickle.dump(wit_dataset, open(f"{project_dir}/wit_dataset.pkl", "wb"))
index_path = f"{project_dir}/wit_faiss_{new_dimension}.idx"
faiss.write_index(index, index_path)

"""
We test now the faiss index with some random entries from the datasets
"""

faiss_index = faiss.read_index(index_path)
descriptions_size = 0
descriptions_test = []
last_dataframe_size = 0

random.seed(10)
for wit_file in sorted(Path(WIT_dir).glob("*.tsv")):
    print(wit_file)
    descriptions, image_info, desc2image_map, dataframe_size = WitDataset.read(wit_file)

    index_test = sorted([randint(0, len(descriptions)-1) for i in range(10)])
    descriptions_test += [[descriptions[i], i+descriptions_size] for i in index_test]
    last_dataframe_size += dataframe_size
    descriptions_size += len(descriptions)

descriptions = [value[0] for value in descriptions_test]
embeddings = model.encode(descriptions, show_progress_bar=True)

for i, value in enumerate(descriptions_test):
    distance, index = faiss_index.search(np.array([embeddings[i]]), k=10)
    print(f"\n### {i:} Search id = {value[1]}")
    print(f'L2 distance: {distance.flatten().tolist()}\nMAG paper IDs: {index.flatten().tolist()}')
    print(f"text: << {value[0]} >>")
    print(f"value: << {value[1]} >>")
    index_url = wit_dataset['desc2image_map'][value[1]]
    print(f"index_url: {index_url}")
    print(f"url: {wit_dataset['image_info'][index_url]}")
    assert (value[1] in index[0])

exit(0)
