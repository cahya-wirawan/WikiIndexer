import re
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from deep_getsizeof import deep_getsizeof


class WitDataset:
    def __init__(self):
        pass

    @staticmethod
    def read(path: Path, lang="en", length=None):
        print(f"Reading the wit_dataset {path}")
        descriptions = []
        desc2image_map = []
        image_info = {}
        dataframe_size = 0
        chunksize = 100000
        with pd.read_csv(path, sep="\t", chunksize=chunksize) as reader:
            for x, chunk in enumerate(reader):
                dataframe_size += len(chunk)
                for i, row in tqdm(chunk.iterrows(), total=len(chunk)):
                    if length is not None and len(descriptions) > length:
                        return descriptions, image_info, desc2image_map, dataframe_size
                    if row["language"] == lang:
                        image_info[i] = [row["image_url"], row["caption_reference_description"]]
                        if type(row["caption_reference_description"]) == str:
                            caption_reference_description = re.sub(r'\s{2,}', " ", row["caption_reference_description"])
                            descriptions.append(caption_reference_description)
                            desc2image_map.append(i)
                        if type(row["context_page_description"]) == str:
                            context_page_description = re.sub(r'\s{2,}', " ", row["context_page_description"])
                            descriptions.append(context_page_description)
                            desc2image_map.append(i)
                            if type(row["context_section_description"]) == str:
                                context_section_description = re.sub(r"\([^)]+\)", "",
                                                                     row["context_section_description"])
                                context_section_description = re.sub(r'\s{2,}', " ", context_section_description)
                                if context_section_description != context_page_description:
                                    descriptions.append(context_section_description)
                                    desc2image_map.append(i)
                        elif type(row["context_section_description"]) == str:
                            context_section_description = re.sub(r'\s{2,}', " ", row["context_section_description"])
                            descriptions.append(context_section_description)
                            desc2image_map.append(i)
                    if i % chunksize == 0:
                        print(f"{x}: Size of desc: {deep_getsizeof(descriptions)/2**20:0.2f} MB, "
                              f"image_info: {deep_getsizeof(image_info)/2**20:0.2f} MB, "
                              f"desc2image_map: {deep_getsizeof(desc2image_map)/2**20:0.2f} MB")
        return descriptions, image_info, desc2image_map, dataframe_size
