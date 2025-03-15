import os
import tqdm
import json
import numpy as np

from PIL import Image
from typing import List
from torch.utils.data import Dataset

from .utils import load_jsonl

INTENSITY_MAP = {
    'not harmful': 0, 
    'somewhat harmful': 1, 
    'very harmful': 1
}

TARGET_MAP = {
    'individual': 0, 
    'organization': 1, 
    'community': 2 , 
    'society': 3
}

class HarmemesBase(Dataset):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str]
    ):  
        self.labels = labels
        self.annotations = self._preprocess_annotations(annotation_filepath)
        self.auxiliary_data = self._load_auxiliary(auxiliary_dicts)
        

    def _preprocess_annotations(self, annotation_filepath: str):
        annotations = []

        # load the default annotations
        data = load_jsonl(annotation_filepath)

        record_id = 0
        
        # translate labels into numeric values
        if "intensity" in self.labels:
            for record in tqdm.tqdm(data, desc="Preprocessing labels"):
                record["img"] = record.pop("image")
                record["intensity"] = INTENSITY_MAP[record["labels"][0]]
                record["id"] = record_id
                record_id += 1
                annotations.append(record)
        
        else:
            for record in tqdm.tqdm(data, desc="Preprocessing labels"):
                record["img"] = record.pop("image")
                record["target"] = TARGET_MAP[record["labels"][1]] if len(record["labels"]) > 1 else -1
                record["id"] = record_id
                record_id += 1
                if record["target"] != -1:
                    annotations.append(record)
        
        return annotations

    def _load_auxiliary(self, auxiliary_dicts: dict):
        data = {}
        for key, filepath in tqdm.tqdm(auxiliary_dicts.items(), desc="Loading auxiliary info"):
            with open(filepath, "r") as f:
                data[key] = json.load(f)

        return data

    def __len__(self):
        return len(self.annotations)


class FasterRCNNDataset(HarmemesBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str],
        feats_dict: dict
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, labels)
        self.feats_dict = feats_dict

    def __getitem__(self, idx: int):
        record = self.annotations[idx]

        text = record['text']
        image_id = record['img']
        id, _ = os.path.splitext(image_id)

        item = {
            'id': id,
            'image_id': image_id,
            'text': text,
            'roi_features': self.feats_dict[id]['roi_features'],
            'normalized_boxes': self.feats_dict[id]['normalized_boxes']
        }

        for l in self.labels:
            item[l] = record[l]

        return item


class ImageDataset(HarmemesBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        text_template: str,
        labels: List[str],
        image_dir: str,
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, labels)
        self.image_dir = image_dir
        self.text_template = text_template

    def __getitem__(self, idx: int):
        record = self.annotations[idx]

        image_filename = record['img']

        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = image.convert("RGB") if image.mode != "RGB" else image

        # text formatting
        input_kwargs = {"text": record['text']}
        for key, data in self.auxiliary_data.items():
            input_kwargs[key] = data[image_filename]

        text = self.text_template.format(**input_kwargs)

        item = {
            'id': record['id'],
            'image_filename': image_filename,
            'text': text,
            'image': np.array(image),
            'image_path': image_path
        }

        for l in self.labels:
            item[l] = record[l]

        return item

class IntMemeDataset(HarmemesBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        text_template: str,
        labels: List[str],
        image_dir: str,
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, labels)
        self.image_dir = image_dir
        self.text_template = text_template

    def __getitem__(self, idx: int):
        record = self.annotations[idx]

        image_filename = record['img']

        image_path = os.path.join(self.image_dir, image_filename)
        image = Image.open(image_path)
        image = image.resize((224, 224))
        image = image.convert("RGB") if image.mode != "RGB" else image

        # text formatting
        input_kwargs = {}
        for key, data in self.auxiliary_data.items():
            input_kwargs[key] = data[image_filename]

        passage = self.text_template.format(**input_kwargs)

        item = {
            'id': record['id'],
            'image_filename': image_filename,
            'text': record['text'],
            'passage': passage,
            'image': np.array(image),
            'image_path': image_path
        }

        for l in self.labels:
            item[l] = record[l]

        return item


class TextDataset(HarmemesBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str],
        input_template: str,
        output_template: str,
        label2word: dict
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, labels)
        self.input_template = input_template
        self.output_template = output_template
        self.label2word = label2word

    def __getitem__(self, idx: int):
        record = self.annotations[idx]

        # Format the input template
        input_kwargs = {"text": record['text']}
        for key, data in self.auxiliary_data.items():
            input_kwargs[key] = data[f"{id:05}"]

        image_id, _ = os.path.splitext(record['img'])

        item = {
            'id': record["id"],
            'image_id': image_id,
            'text': self.input_template.format(**input_kwargs)
        }

        for l in self.labels:
            label = record[l]
            if label == -1:
                continue
            item[l] = self.output_template.format(label=self.label2word[label])

        return item

class TextClassificationDataset(HarmemesBase):
    def __init__(
        self,
        annotation_filepath: str,
        auxiliary_dicts: dict,
        labels: List[str],
        input_template: str
    ):
        super().__init__(annotation_filepath, auxiliary_dicts, labels)
        self.input_template = input_template

    def __getitem__(self, idx: int):
        record = self.annotations[idx]
        image_filename = record['img']

        # Format the input template
        input_kwargs = {"text": record['text']}
        for key, data in self.auxiliary_data.items():
            input_kwargs[key] = data[image_filename]

        image_id, _ = os.path.splitext(record['img'])

        item = {
            'id': record["id"],
            'image_id': image_id,
            'text': self.input_template.format(**input_kwargs)
        }

        for l in self.labels:
            item[l] = record[l]

        return item