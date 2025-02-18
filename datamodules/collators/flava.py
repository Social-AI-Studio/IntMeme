import os
import torch

def image_collate_fn(batch, processor, labels):    
    texts, images = [], []
    img_filenames = []
    for item in batch:
        img_filename, _ = os.path.splitext(item['image_filename'])
        texts.append(item["text"])
        images.append(item["image"])
        # img_filenames.append(int(img_filename))
    
    inputs = processor(  
        text=texts, images=images, return_tensors="pt", padding=True, truncation=True
    )
    # inputs['image_filename'] = torch.tensor(img_filenames, dtype=torch.int64)

    # Get Labels
    for l in labels:
        if l in batch[0].keys():
            labels = [feature[l] for feature in batch]
            inputs[l] = torch.tensor(labels, dtype=torch.int64)

    return inputs