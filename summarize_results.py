import os
import numpy as np
import pandas as pd


def read_file(dataset, base=True, K1=8, K2=8, seed=1):
    dir = "/shared/s2/lab01/myungjoo/RPO_v2/output/rpo_prime/base2new"
    if base:
        dir += '/train_base'
    else:
        dir += '/test_new'

    dir += '/{}'.format(dataset)
    dir += '/shots_16/RPO_prime_sdl'
    dir += '/main_tmp1/seed{}'.format(seed)

    file_path = dir + '/log.txt'

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            read_lines = f.readlines()[-5:]
        # if 'macro_f1' in read_lines[-1]:
        #     accuracy = read_lines[-3].strip().split(' ')[-1]
        #     return float(accuracy[:-1])
        if base:
            if 'macro_f1' in read_lines[-2]:
                accuracy = read_lines[-4].strip().split(' ')[-1]
                return float(accuracy[:-1])
        else:
            if 'macro_f1' in read_lines[-1]:
                accuracy = read_lines[-3].strip().split(' ')[-1]
                return float(accuracy[:-1])
    return None


dataset_list = ['eurosat', 'dtd', 'fgvc_aircraft', 'oxford_flowers', 'stanford_cars', 'oxford_pets', 'food101',
                'sun397', 'ucf101', 'caltech101', 'imagenet']
prompt_list = [[24, 8], [8, 24], [8, 4], [8, 8], [4, 4], [4, 8]]
seed_list = [1, 2, 3]

for K1, K2 in [[0,0]]:
    result = {}
    for dataset in dataset_list:
        result[dataset] = {'base': [], 'novel': []}
        for seed in seed_list:
            base_accr = read_file(dataset, True, K1, K2, seed)
            novl_accr = read_file(dataset, False, K1, K2, seed)
            result[dataset]['base'].append(base_accr if base_accr is not None else 0)
            result[dataset]['novel'].append(novl_accr if base_accr is not None else 0)
        result[dataset] = result[dataset]['base'] + [np.mean(result[dataset]['base'])] + result[dataset]['novel'] + [np.mean(result[dataset]['novel'])]
    df = pd.DataFrame(result).T
    df.to_csv('./results_tmp1_1sdl.csv')








