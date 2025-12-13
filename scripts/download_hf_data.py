import os
import pandas as pd
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import io
from PIL import Image

def download_and_process_hf_data():
    print("Downloading FER2013 from Hugging Face...")
    # Try the parquet version which doesn't need remote code
    try:
        dataset = load_dataset("Jeneral/fer2013", split="train")
    except Exception as e:
        print(f"Could not load 'Jeneral/fer2013': {e}")
        return

    print("Processing dataset...")
    
    emotions = []
    pixels_list = []
    usages = []
    
    print(f"Columns: {dataset.column_names}")
    
    for item in tqdm(dataset):
        # Handle Emotion
        if 'emotion' in item:
            emotions.append(item['emotion'])
        elif 'label' in item:
            emotions.append(item['label'])
        elif 'labels' in item:
            emotions.append(item['labels'])
            
        # Handle Pixels/Image
        if 'pixels' in item:
            p = item['pixels']
            if isinstance(p, list):
                p_str = ' '.join(map(str, p))
            else:
                p_str = p
            pixels_list.append(p_str)
        elif 'image' in item:
            # Convert PIL image to 48x48 grayscale string
            img = item['image']
            if img.mode != 'L':
                img = img.convert('L')
            img = img.resize((48, 48))
            arr = np.array(img).flatten()
            p_str = ' '.join(map(str, arr))
            pixels_list.append(p_str)
        elif 'img_bytes' in item:
             img = Image.open(io.BytesIO(item['img_bytes']))
             if img.mode != 'L':
                img = img.convert('L')
             img = img.resize((48, 48))
             arr = np.array(img).flatten()
             p_str = ' '.join(map(str, arr))
             pixels_list.append(p_str)
        
        # Usage might be present or we infer it
        if 'Usage' in item:
            usages.append(item['Usage'])
        elif 'split' in item:
             usages.append(item['split'])
        else:
            pass

    # If usages are missing, generate them based on index
    if not usages:
        total = len(emotions)
        # Standard FER2013 counts
        n_train = 28709
        n_val = 3589
        n_test = 3589
        
        # If total doesn't match, we just do a random split or 80/10/10
        if total == n_train + n_val + n_test:
            usages = ['Training'] * n_train + ['PublicTest'] * n_val + ['PrivateTest'] * n_test
        else:
            print(f"Dataset size {total} doesn't match standard FER2013. Doing 80/10/10 split.")
            n_train = int(total * 0.8)
            n_val = int(total * 0.1)
            n_test = total - n_train - n_val
            usages = ['Training'] * n_train + ['PublicTest'] * n_val + ['PrivateTest'] * n_test

    df = pd.DataFrame({
        'emotion': emotions,
        'pixels': pixels_list,
        'Usage': usages
    })
    
    output_path = 'data/raw/fer2013.csv'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved dataset to {output_path}")

if __name__ == "__main__":
    download_and_process_hf_data()
