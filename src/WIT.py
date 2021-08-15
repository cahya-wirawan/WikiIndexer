import re
from pathlib import Path
import os
import sys
import pickle
from deep_getsizeof import deep_getsizeof

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
    counter = 0
    with pd.read_csv(path, sep="\t", chunksize=100000) as reader:
        for x, chunk in enumerate(reader):
            for i, row in tqdm(chunk.iterrows(), total=len(chunk)):
                if length is not None and counter > length:
                    break
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
    return descriptions, image_urls, index_map


for wit_file in Path(WIT_dir).glob("*.tsv"):
    print(wit_file)
exit(0)

WIT_file = Path(WIT_dir)/Path(WIT_files[0])
descriptions, image_urls, index_map = WIT_read(WIT_file)
print(len(descriptions))
#exit(0)
wit_dataset = {
    "image_urls": image_urls,
    "index_map": index_map
}
print("saving the wit_dataset")
pickle.dump(wit_dataset, open("wit_dataset.pkl", "wb"))
wit_dataset["descriptions"] = descriptions
if len(sys.argv) == 2:
    new_dimension = int(sys.argv[1])
else:
    new_dimension = 128
model_name = f"distilbert-base-wiki-{new_dimension}"
print(f"Model name: {model_name}")

passages_length = len(descriptions)

device = torch.device("cuda")

if not os.path.exists(model_name):
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

    if torch.cuda.is_available():
        model = model.to(device)
    print(model.device)
    #wiki_snippets_shuffled = wiki_snippets.shuffle(seed=42)
    wiki_sentences = descriptions[:int(0.2*len(descriptions))]
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

batch_size = 16000
steps = passages_length//batch_size
print("Start encoding the passages", passages_length, batch_size, steps)

# Convert abstracts to vectors
embeddings = None
end = 0
for step in tqdm(range(steps)):
    start, end = step*batch_size, (step+1)*batch_size
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
wiki_ids = np.array(range(passages_length))

# Step 1: Change data type
# embeddings = np.array([embedding for embedding in embeddings]).astype("float32")

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

search_id = 30
id = faiss.read_index(index_path)
D, I = id.search(np.array([embeddings[search_id]]), k=10)
print(f'L2 distance: {D.flatten().tolist()}\n\nMAG paper IDs: {I.flatten().tolist()}')
print(f"text: << {descriptions[search_id]} >>\nurl: {image_urls[index_map[search_id]]}")