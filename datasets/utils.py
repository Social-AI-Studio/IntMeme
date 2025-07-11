import json
import csv

def load_jsonl(filepath):
    data = []
    
    with open(filepath, 'r')as f:
        for line in f.readlines():
            obj = json.loads(line)
            data.append(obj)
    
    return data