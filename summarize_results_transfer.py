import os
import numpy as np
import pandas as pd
import pdb

def read_file(source_dataset,dataset,  seed=1):
    dir = "/shared/s2/lab01/myungjoo/RPO_v2/output/rpo_prime/crossdataset_993"
    dir += '/test_target'

    dir += f"/source_{source_dataset}/{dataset}/seed{seed}"

    file_path = dir + '/log.txt'

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            read_lines = f.readlines()[-5:]

        if 'macro_f1' in read_lines[-1]:
            accuracy = read_lines[-3].strip().split(' ')[-1]
            return float(accuracy[:-1]) * 0.01
    return None

source_list = "imagenet caltech101 oxford_pets stanford_cars oxford_flowers food101 fgvc_aircraft sun397 dtd eurosat ucf101".split()
#source_list = "eurosat dtd fgvc_aircraft oxford_flowers stanford_cars oxford_pets food101 ucf101 caltech101 sun397 imagenet".split()

seed_list = [1, 2, 3]

result = {}
for source_dataset in source_list:
    result ={}
    for dataset in source_list:
        #if source_dataset != dataset:
        result[dataset] = {'seed1':0, 'seed2':0, 'seed3':0}
        for seed in seed_list:
            accr = read_file(source_dataset,dataset, seed)
            result[dataset][f'seed{seed}'] = accr if accr is not None else 0
        df = pd.DataFrame(result)
        df.to_csv(f'./transfer_csv_rpoprime_993/source_{source_dataset}.csv')







