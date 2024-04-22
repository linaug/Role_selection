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
    df1 = pd.read_pickle(df1)

    df_explored = df1[(df1['comm'] == False) & (df1['agent'] == 4)]

    df1_ = df1[(df1['comm'] == False) & (df1['agent'] > 0)]

    # calculate the covered ratio
    d_coverage = (df1_.sort_values(['trial', 'step']).groupby(['trial', 'step'])['reward_coverage'].apply("sum", 'step').groupby('trial').cumsum()).groupby('step')
    d_step = d_coverage.mean()/max_cov

    # calculate the explored ratio
    d_explore = (df_explored.sort_values(['trial', 'step']).groupby(['trial', 'step'])['explored_ratio'].apply("mean", 'step').groupby('trial').cumsum()).groupby('step')
    d_explore_step = d_explore.mean()

    # 找覆盖率大于某一个阈值时的时间步和重复覆盖率
    last_cov = d_step[d_step.index[-1]]
    last_explore = d_explore_step[d_explore_step.index[-1]]

    if last_explore >= 0.9:
        explored_ratio = d_explore_step[d_explore_step[:]>0.9][:1]
        out_explored = {'explore_ratio':last_explore,
                        'max_test_step':d_explore_step.index[-1],
                        'step_explored_more_90':explored_ratio.index[0]}
    elif last_explore >= 0.8 and last_explore < 0.9:
        explored_ratio = d_explore_step[d_explore_step[:]>=0.8][:1]
        out_explored = {'explore_ratio':last_explore,
                        'max_test_step':d_explore_step.index[-1],
                        'step_explored_80to90':explored_ratio.index[0]}
    elif last_explore < 0.8:
        out_explored = {'explore_ratio':last_explore,
                        'max_test_step':d_explore_step.index[-1]}
    print(pd.Series(out_explored))

    if last_cov >= 0.8:
        ratio = d_step[d_step[:]>0.8][:1]
        ratio_70 = d_step[d_step[:]>0.7][:1]
        out_covered = {'coverd_ratio':d_step[d_step.index[-1]],
                       'max_test_step':d_step.index[-1],
                       'step_cover_80':ratio.index[0],
                       'step_cover_70to80':ratio_70.index[0]}
    elif last_cov >= 0.7 and last_cov <= 0.8:
        ratio = d_step[d_step[:]>0.7][:1]
        out_covered = {'coverd_ratio':d_step[d_step.index[-1]],
                       'max_test_step':d_step.index[-1],
                       'cover_70to80_step':ratio.index[0]}
    elif last_cov < 0.7:
        
        out_covered = {'coverd_ratio':d_step[d_step.index[-1]],
                       'max_test_step':d_step.index[-1]}

    print(pd.Series(out_covered))
    print("-"*60)

if __name__ == '__main__':
    # end2end with 4 robots
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/end2end/end2end-robots_4-interst_0.56-coverable_0.6-eval_coop-checkpoint-.pkl"
    # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/end2end/end2end-robots_4-interst_0.24-coverable_0.6-eval_coop-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/end2end/end2end-robots_4-interst_0.16-coverable_0.6-eval_coop-checkpoint-.pkl"
    # df4 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/end2end/end2end-robots_4-interst_0.08-coverable_0.6-eval_coop-checkpoint-.pkl"

    # # end2end with 15 robots
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/end2end/end2end-robots_15-interst_0.56-coverable_0.6-eval_coop-checkpoint-.pkl"
    # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/end2end/end2end-robots_15-interst_0.24-coverable_0.6-eval_coop-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/end2end/end2end-robots_15-interst_0.16-coverable_0.6-eval_coop-checkpoint-.pkl"
    # df4 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/end2end/end2end-robots_15-interst_0.08-coverable_0.6-eval_coop-checkpoint-.pkl"

    # end2end with 8 robots
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/end2end/end2end-robots_8-interst_0.56-coverable_0.6-eval_coop-checkpoint-.pkl"
    # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/end2end/end2end-robots_8-interst_0.24-coverable_0.6-eval_coop-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/end2end/end2end-robots_8-interst_0.16-coverable_0.6-eval_coop-checkpoint-.pkl"
    # df4 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/end2end/end2end-robots_8-interst_0.08-coverable_0.6-eval_coop-checkpoint-.pkl"
    
    # # end2end with different obstacles
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/end2end/end2end-robots_4-interst_0.16-coverable_0.84-eval_coop-checkpoint-.pkl"
    # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/end2end/end2end-robots_4-interst_0.16-coverable_0.6-eval_coop-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/end2end/end2end-robots_4-interst_0.16-coverable_0.48-eval_coop-checkpoint-.pkl"

    # # Random with 4 robots
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/random/Random-interst_0.56-coverable_0.6-robots_4-0.4explore_0.6cover-checkpoint-.pkl"
    # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/random/Random-interst_0.24-coverable_0.6-robots_4-0.4explore_0.6cover-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/random/Random-interst_0.16-coverable_0.6-robots_4-0.4explore_0.6cover-checkpoint-.pkl"
    # df4 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/random/Random-interst_0.08-coverable_0.6-robots_4-0.4explore_0.6cover-checkpoint-.pkl"

    # # Random with 8 robots
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/random/Random-interst_0.56-coverable_0.6-robots_8-0.4explore_0.6cover-checkpoint-.pkl"
    # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/random/Random-interst_0.24-coverable_0.6-robots_8-0.4explore_0.6cover-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/random/Random-interst_0.16-coverable_0.6-robots_8-0.4explore_0.6cover-checkpoint-.pkl"
    # df4 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/random/Random-interst_0.08-coverable_0.6-robots_8-0.4explore_0.6cover-checkpoint-.pkl"

    # # Random with 15 robots
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/random/Random-interst_0.56-coverable_0.6-robots_15-0.4explore_0.6cover-checkpoint-.pkl"
    # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/random/Random-interst_0.24-coverable_0.6-robots_15-0.4explore_0.6cover-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/random/Random-interst_0.16-coverable_0.6-robots_15-0.4explore_0.6cover-checkpoint-.pkl"
    # df4 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/random/Random-interst_0.08-coverable_0.6-robots_15-0.4explore_0.6cover-checkpoint-.pkl"

    # # Random with 4 robots and different obstacles
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/random/Random-interst_0.16-coverable_0.84-robots_4-0.4explore_0.6cover-checkpoint-.pkl"
    # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/random/Random-interst_0.16-coverable_0.6-robots_4-0.4explore_0.6cover-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/random/Random-interst_0.16-coverable_0.49-robots_4-0.4explore_0.6cover-checkpoint-.pkl"

    # Greedy with
    df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/greedy/Greedy-interst_0.16-coverable_0.84-robots_4-checkpoint-.pkl"
    df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/greedy/Greedy-interst_0.16-coverable_0.48-robots_4-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/Greedy/Greedy-robots_4-interst_0.16-coverable_0.48-eval_coop-checkpoint-.pkl"
    # df4 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/Greedy/Greedy-robots_8-interst_0.08-coverable_0.6-eval_coop-checkpoint-.pkl"
    # df = [df1]
    
    # role-based with 4 robots
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/robots_4/eval_coop-interst_0.56-coverable_0.6-robots_4-0.4explore_0.6cover-checkpoint-.pkl"
    # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/robots_4/eval_coop-interst_0.24-coverable_0.6-robots_4-0.4explore_0.6cover-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/robots_4/eval_coop-interst_0.16-coverable_0.6-robots_4-0.4explore_0.6cover-checkpoint-.pkl"
    # df4 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/robots_4/eval_coop-interst_0.08-coverable_0.6-robots_4-0.4explore_0.6cover-checkpoint-.pkl"
    # df = [df1,df2,df3,df4]

    # role-based with 8 robots
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/robots_8/eval_coop-interst_0.56-coverable_0.6-robots_8-0.4explore_0.6cover-checkpoint-.pkl"
    # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/robots_8/eval_coop-interst_0.24-coverable_0.6-robots_8-0.4explore_0.6cover-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/robots_8/eval_coop-interst_0.16-coverable_0.6-robots_8-0.4explore_0.6cover-checkpoint-.pkl"
    # df4 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/robots_8/eval_coop-interst_0.08-coverable_0.6-robots_8-0.4explore_0.6cover-checkpoint-.pkl"
    

    # role-based with 15 robots
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/robots_15/eval_coop-interst_0.56-coverable_0.6-robots_15-0.4explore_0.6cover-checkpoint-.pkl"
    # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/robots_15/eval_coop-interst_0.24-coverable_0.6-robots_15-0.4explore_0.6cover-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/robots_15/eval_coop-interst_0.16-coverable_0.6-robots_15-0.4explore_0.6cover-checkpoint-.pkl"
    # df4 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/robots_15/eval_coop-interst_0.08-coverable_0.6-robots_15-0.4explore_0.6cover-checkpoint-.pkl"
    
    # role-based with 4 robots with different obstacles
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/obstacles_easy2hard/eval_coop-interst_0.16-coverable_0.84-robots_4-0.4explore_0.6cover-checkpoint-.pkl"
    # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/obstacles_easy2hard/eval_coop-interst_0.16-coverable_0.6-robots_4-0.4explore_0.6cover-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/obstacles_easy2hard/eval_coop-interst_0.16-coverable_0.49-robots_4-0.4explore_0.6cover-checkpoint-.pkl"
    
    df = [df1,df2]
    # max_cov = [0.56*25*25,0.24*25*25,0.16*25*25,0.08*25*25]
    max_cov = [0.16*25*25,0.16*25*25]
    for i in range(len(df)):
        # aa = df[i][-50:-16]
        aa = df[i][-72:-16]
        print("In test model:",aa)
        output_results(df[i],max_cov[i])




    
