import pandas as pd
import os
from translator_gpt import openai_translator as zhvi_translator

"""
formating:
    (1) index: int
    (2) start timestamp (hh:mm:ss,ms) --> end timestamp (hh:mm:ss,ms)
    (3) content: string
    (4) space \n
"""
ROOT = 'C:\BANGLV\capcut\ep3'
path = os.path.join(ROOT, 'infors_filtered.csv')
to_save = os.path.join(ROOT, 'infors_filtered.txt')
df = pd.read_csv(path)
infors = df.values.tolist()

f = open(os.path.join(ROOT, "results/subtitle.txt"), "w", encoding='utf-8-sig')
for [c, st, et, tx] in infors:
    print(c, st, et, tx)
    # f.write(f'{c} \n')
    ted = zhvi_translator(tx)
    print(ted, '\n')
    f.write(f'{st} --> {et} \n')
    f.write(f'{tx} \n')
    f.write(f'{ted} \n')
    f.write(' \n')

f.close()