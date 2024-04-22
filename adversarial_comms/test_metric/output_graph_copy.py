import argparse
from jsonschema import draft201909_format_checker 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb

def plot_agent_explore(ax, df, color, linestyle='-',linewidth=2):
    world_shape = df.attrs['env_config']['world_shape']
    d = df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['explored_ratio'].apply('mean', 'step').groupby('step').apply('mean','trial').groupby('step')
    ax.plot(d.mean()/0.8831, color=color, ls=linestyle, linewidth=linewidth)
   # ax.plot(d.mean(), color=color, ls=linestyle, linewidth=linewidth)
    # ax.fill_between(np.arange(len(d.mean())), np.clip(d.mean()-d.std(), 0, None), d.mean()+d.std(), alpha=0.1, color=color)

def plot_agent_loudian(ax, df, color, linestyle='-',linewidth=2):
    world_shape = df.attrs['env_config']['world_shape']
    d_ = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['loudian_ratio'].apply('mean', 'step').groupby('step').apply('mean','trial')).groupby('step')
    return d_.mean()
    # ax.plot(d_.mean(), color=color, ls=linestyle, linewidth=linewidth)
    # ax.fill_between(np.arange(len(d.mean())), np.clip(d.mean()-d.std(), 0, None), d.mean()+d.std(), alpha=0.1, color=color)

def plot_agent_coverage(ax, df, color, step_aggregation='sum', linestyle='-'):
    world_shape = df.attrs['env_config']['world_shape']
    max_cov = 0.16
    d = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['reward_coverage'].apply(step_aggregation, 'step').groupby('trial').cumsum()).groupby('step')
    ax.plot(d.mean()/100, color=color, ls=linestyle)
    # ax.fill_between(np.arange(len(d.mean())), np.clip(d.mean()-d.std(), 0, None), d.mean()+d.std(), alpha=0.1, color=color)

def plot_loudian_coverage(df1):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_file", default=None)
    args = parser.parse_args()

    fig_overview = plt.figure(figsize=[4, 4])
    ax = fig_overview.subplots(1, 1)

    df1 = pd.read_pickle(df1)

    # varify_explored = True
    # varify_loudian = False
  
    x = plot_agent(ax ,df1[(df1['comm'] == False) & (df1['agent'] > 0)], 'y', linestyle='-')
    
    y = plot_agent_loudian(ax ,df1[(df1['comm'] == False) & (df1['agent'] > 0)], 'y', linestyle=':')

    line1, = ax.plot([], [], color='y', linestyle='-')

    # # line4, = ax.plot([], [], color='grey', linestyle='-')
   
    ax.set_xlabel("Explored ratio %")
    ax.set_ylabel("Forgetting ratio %")

    # ax.legend(handles=[line,line_1,line_01,line_02,line_03,line_04,line1,line2,line3], labels=['Role-A_1-B_0',
    #                                                     'Role-A_0-B_1',
    #                                                     'Role-A_0.1-B_0.9',
    #                                                     'Role-A_0.2-B_0.8',
    #                                                     'Role-A_0.3-B_0.7',
    #                                                     'Role-A_0.4-B_0.6',
    #                                                     'End2end',"Random","Greedy"], loc='best')
    # ax.legend(handles=[line_01], labels=['Role-A_0.1-B_0.9'], loc='best')
    plt.plot(x,y, color='r', ls='-', linewidth=2)
    ax.margins(x=0, y=0)
    ax.grid()
    fig_overview.tight_layout()
    if args.out_file is not None:
        fig_overview.savefig(args.out_file, dpi=300)

    plt.show()

def plot_(df1,df2):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_file", default=None)
    args = parser.parse_args()

    fig_overview = plt.figure(figsize=[4, 4])
    ax = fig_overview.subplots(1, 1)

    df1 = pd.read_pickle(df1)
    df2 = pd.read_pickle(df2)
    # df3 = pd.read_pickle(df3)
    # df4 = pd.read_pickle(df4)

    # varify_explored = True
    # varify_loudian = False
  
    # plot_agent_coverage(ax ,df1[(df1['comm'] == False) & (df1['agent'] > 0)], 'y', linestyle='-')
    # plot_agent_coverage(ax ,df2[(df2['comm'] == False) & (df2['agent'] > 0)], 'r', linestyle='-')
    # plot_agent_coverage(ax ,df3[(df3['comm'] == False) & (df3['agent'] > 0)], 'g', linestyle='-')
    # plot_agent_coverage(ax ,df4[(df4['comm'] == False) & (df4['agent'] > 0)], 'black', linestyle='-')
  
    plot_agent_explore(ax ,df1[(df1['comm'] == False) & (df1['agent'] > 0)], 'y', linestyle='-')
    plot_agent_explore(ax ,df2[(df2['comm'] == False) & (df2['agent'] > 0)], 'r', linestyle='-')
    # plot_agent_explore(ax ,df3[(df3['comm'] == False) & (df3['agent'] > 0)], 'g', linestyle='-')
    # plot_agent_explore(ax ,df4[(df4['comm'] == False) & (df4['agent'] > 0)], 'black', linestyle='-')
    # plot_agent_coverage(ax ,df1[(df1['agent'] > 0)], 'y', linestyle='-')
    
    line1, = ax.plot([], [], color='y', linestyle='-')
    line2, = ax.plot([], [], color='r', linestyle='-')  
    # line3, = ax.plot([], [], color='g', linestyle='-')
    # line4, = ax.plot([], [], color='black', linestyle='-')
    
    ax.set_ylabel("Covered ratio %")
    ax.set_ylim(0,1)
    ax.set_xlabel("Episode time steps(Radiu_exploratio_2)")
    ax.set_xlim(0,50)

    # ax.legend(handles=[line1,line2,line3,line4], labels=['Role-A_0-B-1','Role-A_0.2-B-0.8','Role-A_0.3-B_0.7','end2end'], loc='best')
    ax.legend(handles=[line1,line2], labels=['Role-A_0.3-B_0.7','Role-A_1-B_0'], loc='best')
    ax.margins(x=0, y=0)
    ax.grid()
    fig_overview.tight_layout()
    if args.out_file is not None:
        fig_overview.savefig(args.out_file, dpi=300)

    plt.show()

if __name__ == '__main__':
   
    # df1 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/end2end_eval_coop-checkpoint-.pkl" #end2end
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/Greedy_eval_coop-checkpoint-.pkl" #end2end
    # df1 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/greedy/greedy_eval_coop-checkpoint-.pkl" #end2end
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/greedy/greedy_eval_coop-checkpoint-.pkl"
    # df1 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/greedy/eval_coop-checkpoint-.pkl"

    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/eval_coop-0.0explore_1.0cover-checkpoint-.pkl"
    # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.0explore_1.0cover-checkpoint-.pkl"

    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/eval_coop-0.0explore_1.0cover-checkpoint-.pkl"
    # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/eval_coop-0.2explore_0.8cover-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/eval_coop-0.3explore_0.7cover-checkpoint-.pkl"
    # df4 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/end2end_eval_coop-checkpoint-.pkl"

    # df3 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/end2end_eval_coop-checkpoint-.pkl" #end2end

    # # train_ontrain
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/not_train_/eval_coop-0.3explore_0.7cover-checkpoint-.pkl"
    # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/eval_coop-0.3explore_0.7cover-checkpoint-.pkl"

    # train_ontrain
    df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/eval_coop-0.3explore_0.7cover-checkpoint-.pkl"
    df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/eval_coop-1.0explore_0.0cover-checkpoint-.pkl"


    # plot_(df1,df2,df3,df4) # 无自私机器人评估
    plot_(df1,df2) # 无自私机器人评估
