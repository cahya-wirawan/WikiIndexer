import faiss
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer, models
import datasets
import pandas as pd
from tqdm import tqdm
import numpy as np
import pickle

# content of test_wit_indexer.py

class TestWitIndexer:
    index_path = "wiki_faiss_128.idx"
    model_name = "distilbert-base-wiki-128"

    def test_one(self):
        x = "this"
        assert "h" in x

    def test_two(self):
        x = "hello"
        assert hasattr(x, "check")

    def test_faiss_index(self):
        faiss_index = faiss.read_index(self.index_path)
        #res = faiss.StandardGpuResources()
        #faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
        model = SentenceTransformer(self.model_name)
        wit_dataset = pickle.load(open("wit_dataset.pkl", "rb"))
        device = torch.device("cuda")
        model = model.to(device)
        descriptions = ["Jakarta is capital of Indonesia ", "Siamese cat"]
        embeddings = model.encode(descriptions)
        for i, value in enumerate(descriptions):
            D, I = faiss_index.search(np.array([embeddings[i]]), k=10)
            D = D.flatten().tolist()
            I = I.flatten().tolist()
            print(f"\n### {i:} Search for {value}")
            print(f'L2 distance: {D[0]}\nMAG paper IDs: {I[0]}')
            index_url = wit_dataset['index_map'][I[0]]
            print(f"index_url: {index_url}")
            print(f"url: {wit_dataset['image_urls'][index_url]}")
            assert D[0] < 100.0