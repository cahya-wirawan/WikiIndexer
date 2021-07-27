import io
import uvicorn
from transformers import pipeline
from pydantic import BaseModel
from fastapi import FastAPI
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from wiki_index import WikiIndex


class Input(BaseModel):
    text: str
    k: int
    include_urls: bool
    weighted_passage: bool


class TextGenResponse(BaseModel):
    result: str


app = FastAPI(
    title="Wiki Search",
    version="0.1.0",
)

wiki_index_file = "/models/wiki/wiki_faiss_128.idx"
model_name = "/models/wiki/distilbert-base-wiki-128"

wiki = WikiIndex(wiki_index_file, model_name)


@app.get('/')
def get_root():
    return {'message': 'Wiki Search'}


@app.post('/search/', response_model=TextGenResponse)
def search(item: Input):
    result = wiki.search(item.text,
                         k=item.k,
                         include_urls=item.include_urls,
                         weighted_passage=item.weighted_passage)


    return {'result': text}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
