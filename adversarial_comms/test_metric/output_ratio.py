from crypt import methods
from json.tool import main
from pathlib import Path
import numpy as np
import argparse
import pandas as pd
import json
import os
from pathlib import Path
import pdb

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def load_txt():
    filepath = '/home/zln/adv_results/5channel/env0/small_5robtos/baseline_2new/eval_coop-checkpoint-.txt'
    data = np.loadtxt(filepath,skiprows=1,usecols=(1,2,3,4,5),encoding='utf-8')
    data_comm = np.loadtxt(filepath,skiprows=1,usecols=(6),dtype=str)

    rows_comm = list(data_comm).index('True')
    data_ = data[rows_comm:,:]
    data_ = data_[0:5,:]
    print(data_)

    for i in range(1):
        data_epcho = np.where(data_[:,2]==i)
        print(len(data_epcho))

def output_results(df1,max_cov):

    df = df1[(df1['comm'] == True) & (df1['agent'] > 0)]

    d_coverage = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['reward'].apply("sum", 'step').groupby('trial').cumsum()/max_cov*100).groupby('step')

    d_step = d_coverage.mean()

    # 找覆盖率大于某一个阈值时的时间步和重复覆盖率
    # out = {'method':methods,'step':ratio.index[0],'duplicate_ration':d_duplicate_[ratio.index[0]]}
    last_cov = d_step[d_step.index[-1]]
    # pdb.set_trace()

    if last_cov >= 95:
        ratio = d_step[d_step[:]>95][:1]
        out = {'step':d_step.index[-1],
                'cover_95_step':ratio.index[0],
                'coverable':d_step[d_step.index[-1]]}
    elif last_cov >= 90 and last_cov <= 95:
        ratio = d_step[d_step[:]>85][:1]

        out = {'step':d_step.index[-1],
                'cover_90to95_step':ratio.index[0],
                'coverable':d_step[d_step.index[-1]]}
    elif last_cov < 90:
        
        out = {'step':d_step.index[-1],
                'coverable':d_step[d_step.index[-1]]}

    print(pd.Series(out))
    print("-"*60)


def output_explored_and_loudian_ratio():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    # parser.add_argument("data2")
    args = parser.parse_args()

    df = pd.read_pickle(args.data)
    # df2 = pd.read_pickle(args.data2)

    max_cov = 0.24 * 25 *25

    output_results(df,max_cov)
    check_(df)


def check_(df):
    # filepath = '/home/zln/adv_results/search_rescue/0717_model_add_loudian/eval_coop-checkpoint-.pkl'
    # df = pd.read_pickle(filepath)
    df = df[(df['comm'] == True) & (df['agent'] > 0)]
    dff = df[(df['step'] == 149) & (df['agent'] == 4)]
    # print(dff)
    explored_avg = dff['explored_ratio'].mean()
    loudian_ratio_avg = dff['loudian_ratio'].mean()

    print('explored reach {} and loudian_ratio is {}'.format(explored_avg,loudian_ratio_avg))


if __name__ == '__main__':
    output_explored_and_loudian_ratio()
    # check_()




    
