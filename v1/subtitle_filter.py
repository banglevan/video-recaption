import pandas as pd
import numpy as np

path = 'results/infors.csv'

df = pd.read_csv(path)
infors = df.values.tolist()
idxs = [i[0] for i in infors]
idxs = list(np.unique(np.array(idxs)))
processed = []
for c in idxs:
    arr = [i for i in infors if i[0] == c]
    s = arr[0][1]
    o = arr[0][3]
    if len(arr) > 1:
        e = arr[-1][2]
        # t = arr[0][4]
    else:
        e = s
    print(c, s, e, o)
    processed.append([c, s, e, o])

rdf = pd.DataFrame(processed, columns=['n_cap', 'stime', 'etime', 'content'])
rdf.to_csv('results/infors_filtered.csv', encoding='utf_8_sig', index=False)