import pandas as pd
import os
from tqdm import tqdm

if __name__ == '__main__':
    exps_path = '../output/log/'
    exps = os.listdir(exps_path)

    benchmark_table = pd.DataFrame()
    columns = ['exp', 'step', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.3:0.7', '0.1:0.7'] 
    for exp in tqdm(exps):
        path = os.path.join(exps_path, exp)
        cur_exp_id = path.split('/')[-1]

        cur_info = []
        with open(os.path.join(path, 'Performance.txt'), 'r') as f:
            test_info = f.readlines()

        for line in test_info[-3:]:
            cur_info.append(line.strip())
        try:
            info_performance = cur_info[0].split('  ')
        except:
            continue
        
        info_performance.insert(0, cur_exp_id)
        max_mAP_row = pd.DataFrame(info_performance, columns)
        
        benchmark_table = pd.concat([benchmark_table, max_mAP_row], axis=1)
        
    benchmark_table = benchmark_table.transpose().reset_index()
    benchmark_table = benchmark_table.sort_values(by='0.1:0.7',ascending=False)
    for idx in range(len(benchmark_table)):
        print(benchmark_table.iloc[idx]['0.3:0.7'] , benchmark_table.iloc[idx]['0.1:0.7'])
    import pdb;pdb.set_trace()


    # benchmark_table.to_csv(f'./benchmark.csv', index=False)
