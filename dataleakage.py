import pandas as pd
import tqdm
import numpy as np
import os

train = pd.read_csv('seti-breakthrough-listen/train_labels.csv')
test = pd.read_csv('seti-breakthrough-listen/sample_submission.csv')

def get_file_path(image_id: str, train_or_test: str, data_dir: str = 'seti-breakthrough-listen/') -> str:
    return os.path.join(data_dir, train_or_test, image_id[0], f"{image_id}.npy")

train['type'] = 'train'
test['type'] = 'test'
df = pd.concat([train, test]).reset_index(drop=True)

mean = []
std = []
minv = []
maxv = []
nuniq = []

for i, row in tqdm(df.iterrows()):
    arr = np.load(get_file_path(row.id, row.type))[[0,2,4],:,:].astype(float)
    mean.append(np.mean(arr))
    std.append(np.std(arr))
    minv.append(np.min(arr))
    maxv.append(np.max(arr))
    nuniq.append(len(np.unique(arr)))
    
df['mean'] = mean
df['std'] = std
df['min'] = minv
df['max'] = maxv
df['nuniq'] = nuniq

df.to_csv('seti-stats.csv', index=False)