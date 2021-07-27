# Used to retrieve the image from Wikipedia article
import requests
from bs4 import BeautifulSoup
import json
import re

# Used to create the dense document vectors.
import torch
from sentence_transformers import SentenceTransformer, models
import datasets

# Used to create and store the Faiss index.
import faiss
import numpy as np


class WikiIndex:
    """
    WikiIndex is a class to search the wiki snippets from the given text. It can also return link to the
    wiki page or the image.
    """
    headers = {
        'cache-control': 'max-age=0',
        'user-agent': 'Mozilla/5.0',
        'accept': 'text/html,application/xhtml+xml,application/xml,application/json',
        'accept-encoding': 'gzip, deflate, br',
        'accept-language': 'en,de;q=0.9,en-US;q=0.8,id;q=0.7'
    }
    wiki_snippets = None

    def __init__(self, index_path, model_path, gpu=True):
        self.index = faiss.read_index(index_path)
        self.model = SentenceTransformer(model_path)
        if WikiIndex.wiki_snippets is None:
            WikiIndex.wiki_snippets = datasets.load_dataset("wiki_snippets", "wiki40b_en_100_0")
        if gpu and torch.cuda.is_available():
            self.model = self.model.to(torch.device("cuda"))
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

    @staticmethod
    def get_url(wiki_id):
        try:
            url = f"https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&props=sitelinks&sitefilter=enwiki&ids={wiki_id}"
            req = requests.get(url, headers=WikiIndex.headers)
            entities = json.loads(req.content)["entities"]
            page = entities[wiki_id]['sitelinks']['enwiki']['title'].replace(" ", "_")
            url = f"https://en.wikipedia.org/wiki/{page}"
        except KeyError:
            url = ""
        return url

    def search(self, text, k=5, include_urls=False, weighted_passage=True):
        embedding = self.model.encode(text, convert_to_numpy=True, show_progress_bar=True)
        # Retrieve the 10 nearest neighbours
        D, I = self.index.search(np.array([embedding]), k=k)
        D, I = D.flatten().tolist(), I.flatten().tolist()
        if weighted_passage:
            passage_index = {I[i]:d for i,d in enumerate(D)}
            for index in passage_index:
                if WikiIndex.wiki_snippets['train'][index]["start_character"] == 0:
                    passage_index[index] = 0.5*passage_index[index]
            passage_index = sorted(passage_index.items(), key=lambda item: item[1])
            D = [pi[1] for pi in passage_index]
            I = [pi[0] for pi in passage_index]
        if include_urls:
            urls = []
            for index in I:
                wiki_id = WikiIndex.wiki_snippets['train'][index]["wiki_id"]
                urls.append(self.get_url(wiki_id))
            return D, I, urls
        else:
            return D, I

    def get_image_url(self, text, index=0, image_width=400):
        D, I = self.search(text)
        image_url = None
        try:
            for index in I:
                wiki_id = WikiIndex.wiki_snippets['train'][index]["wiki_id"]
                url = f"https://www.wikidata.org/wiki/{wiki_id}"
                req = requests.get(url, headers=WikiIndex.headers)
                soup = BeautifulSoup(req.content, 'html.parser')
                image_url = soup.find(property="og:image")
                if image_url is None:
                    continue
                else:
                    image_url = soup.find(property="og:image")["content"]
                    image_url = re.sub(r'/\d*px-', f"/{image_width}px-", image_url)
                    break
        except TypeError:
            image_url = None
        return image_url
