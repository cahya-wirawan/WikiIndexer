import random
import re
from pathlib import Path
import os
import sys
import pickle
from deep_getsizeof import deep_getsizeof
from random import randint, seed

# Used to create the dense document vectors.
import torch
from sentence_transformers import SentenceTransformer, models
import datasets
import pandas as pd
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

def WIT_read(path: Path, lang="en", length=None):
    print(f"Reading the wit_dataset {path}")
    #print(f"Size of pandas dataframe: {getsizeof(df)}")
    descriptions = []
    index_map = []
    image_urls = {}
    dataframe_size = 0
    chunksize = 100
    with pd.read_csv(path, sep="\t", chunksize=chunksize) as reader:
        for x, chunk in enumerate(reader):
            dataframe_size += len(chunk)
            for i, row in tqdm(chunk.iterrows(), total=len(chunk)):
                if length is not None and len(descriptions) > length:
                    return descriptions, image_urls, index_map, dataframe_size
                if row["language"] == lang:
                    image_urls[i] = row["image_url"]
                    if type(row["caption_reference_description"]) == str:
                        caption_reference_description = re.sub(r'\s{2,}', " ", row["caption_reference_description"])
                        descriptions.append(caption_reference_description)
                        index_map.append(i)
                    if type(row["context_page_description"]) == str:
                        context_page_description = re.sub(r'\s{2,}', " ", row["context_page_description"])
                        descriptions.append(context_page_description)
                        index_map.append(i)
                        if type(row["context_section_description"]) == str:
                            context_section_description = re.sub(r'\s{2,}', " ", row["context_section_description"])
                            if context_section_description != context_page_description:
                                descriptions.append(context_section_description)
                                index_map.append(i)
                    elif type(row["context_section_description"]) == str:
                        context_section_description = re.sub(r'\s{2,}', " ", row["context_section_description"])
                        descriptions.append(context_section_description)
                        index_map.append(i)
                if i % 100000 == 0:
                    print(f"{x}: Size of desc: {deep_getsizeof(descriptions)/2**20:0.2f} MB, "
                          f"image_urls: {deep_getsizeof(image_urls)/2**20:0.2f} MB, "
                          f"index_map: {deep_getsizeof(index_map)/2**20:0.2f} MB")
    return descriptions, image_urls, index_map, dataframe_size


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
        del (pca_train_sentences)
        del (wiki_sentences)
    else:
        model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            model = model.to(device)

    return model


wit_dataset = {
    "image_urls": {},
    "index_map": []
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
batch_size = 64
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
    descriptions, image_urls, index_map, dataframe_size = WIT_read(wit_file)
    image_urls = {key+last_dataframe_size:image_urls[key] for key in image_urls}
    wit_dataset["image_urls"] = {**wit_dataset["image_urls"], **image_urls}
    index_map = [i+last_dataframe_size for i in index_map]
    wit_dataset["index_map"] = wit_dataset["index_map"] + index_map
    last_dataframe_size += dataframe_size
    passages_length = len(descriptions)
    if wit_counter == 0:
        model = create_sentence_model(model_name)

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
index_path = f"{project_dir}/wiki_faiss_{new_dimension}.idx"
faiss.write_index(index, index_path)


faiss_index = faiss.read_index(index_path)
descriptions_size = 0
descriptions_test = []
last_dataframe_size = 0

random.seed(1000)
for wit_file in sorted(Path(WIT_dir).glob("*.tsv")):
    print(wit_file)
    descriptions, image_urls, index_map, dataframe_size = WIT_read(wit_file)

    index_test = sorted([randint(0, len(descriptions)) for i in range(10)])
    descriptions_test += [[descriptions[i], i+descriptions_size] for i in index_test]
    last_dataframe_size += dataframe_size
    descriptions_size += len(descriptions)

descriptions = [value[0] for value in descriptions_test]
embeddings = model.encode(descriptions, show_progress_bar=True)

for i, value in enumerate(descriptions_test):
    D, I = faiss_index.search(np.array([embeddings[i]]), k=10)
    print(f"\n### {i:} Search id = {value[1]}")
    print(f'L2 distance: {D.flatten().tolist()}\nMAG paper IDs: {I.flatten().tolist()}')
    print(f"text: << {value[0]} >>")
    print(f"value: << {value[1]} >>")
    index_url = wit_dataset['index_map'][value[1]]
    print(f"index_url: {index_url}")
    print(f"url: {wit_dataset['image_urls'][index_url]}")
    assert (I[0][0] == value[1] or I[0][1] == value[1])

exit(0)
