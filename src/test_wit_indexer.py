import faiss
from pathlib import Path
import torch
from sentence_transformers import SentenceTransformer, models
import datasets
import pandas as pd
from tqdm import tqdm

# content of test_wit_indexer.py

class TestWitIndexer:
    index_path = ""
    model_name = "distilbert-base-nli-stsb-mean-tokens"

    def __init__(self):
        faiss_index = faiss.read_index(self.index_path)
        model = SentenceTransformer(self.model_name)
        
    def test_one(self):
        x = "this"
        assert "h" in x

    def test_two(self):
        x = "hello"
        assert hasattr(x, "check")

    def test_faiss_index(self):
        descriptions_size = 0
        descriptions_test = []
        last_dataframe_size = 0