import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from typing import List, Optional
from wiki_index import WikiIndex


class SearchInput(BaseModel):
    text: str
    k: int
    include_urls: bool
    weighted_passage: bool


class ImageInput(BaseModel):
    text: str
    image_width: int


class SearchResponse(BaseModel):
    probabilities: List[float] = [],
    passage_index: List[int] = [],
    urls: List[str] = []


class ImageResponse(BaseModel):
    url: str

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


@app.post('/search/', response_model=SearchResponse)
def search(item: SearchInput):
    result = wiki.search(item.text,
                         k=item.k,
                         include_urls=item.include_urls,
                         weighted_passage=item.weighted_passage)
    return {
        'probabilities': result[0],
        'passage_index': result[1],
        'urls': result[2]
    }

@app.post('/get_image/', response_model=ImageResponse)
def search(item: ImageInput):
    result = wiki.get_image(item.text, image_width=item.image_width)
    return {
        'url': result
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
