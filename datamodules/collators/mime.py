import os
import torch

def image_collate_fn(batch, processor, tokenizer, labels):    
    texts, images, passages = [], [], []
    img_filenames = []
    for item in batch:
        img_filename, _ = os.path.splitext(item['image_filename'])
        img_filename = ''.join(filter(str.isdigit, img_filename))
        texts.append(item["text"])
        images.append(item["image"])
        passages.append(item["passage"])
        img_filenames.append(int(img_filename))
    
    multimodal_inputs = processor(  
        text=texts, images=images, return_tensors="pt", padding=True, truncation=True
    )

    passage_inputs = tokenizer(  
        text=passages, return_tensors="pt", padding=True, truncation=True
    )

    inputs = {
        "image_filename": torch.tensor(img_filenames, dtype=torch.int64),
        "meme_input_ids": multimodal_inputs.input_ids,
        "meme_attention_mask": multimodal_inputs.attention_mask,
        "pixel_values": multimodal_inputs.pixel_values,
        "passage_input_ids": passage_inputs.input_ids,
        "passage_attention_mask": passage_inputs.attention_mask,
    }

    # Get Labels
    for l in labels:
        if l in batch[0].keys():
            labels = [feature[l] for feature in batch]
            inputs[l] = torch.tensor(labels, dtype=torch.int64)

    return inputs