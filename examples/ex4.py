import numpy as np

txts = ['a']
ptxs = [12]

txts = np.array(txts)
idxs = np.argsort(ptxs)
stxts = list(txts[idxs])

res = "".join(x[0] for x in stxts)
print(res)