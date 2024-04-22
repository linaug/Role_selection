import argparse
from jsonschema import draft201909_format_checker 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb

def plot_agent_loudian(ax, df, color, linestyle='-',linewidth=2):
    world_shape = df.attrs['env_config']['world_shape']
    d_ = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['covered_ratio'].apply('mean', 'step').groupby('step').apply('mean','trial')).groupby('step')
    ax.plot(d_.mean(), color=color, ls=linestyle, linewidth=linewidth)

def plot_agent_role(ax, df, color, linestyle='-',linewidth=2):
    world_shape = df.attrs['env_config']['world_shape']
    d_ = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['role_cover_ratio'].apply('mean', 'step').groupby('step').apply('mean','trial')).groupby('step')
    ax.plot(d_.mean(), color=color, ls=linestyle, linewidth=linewidth)

def plot_agent_role_explore(ax, df, color, linestyle='-',linewidth=2):
    world_shape = df.attrs['env_config']['world_shape']
    d_ = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['role_cover_ratio'].apply('mean', 'step').groupby('step').apply('mean','trial')).groupby('step')
    ax.plot(1-d_.mean(), color=color, ls=linestyle, linewidth=linewidth)


def plot_role(df1,df2,df3,df4):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_file", default="/home/zln/repos/adversarial_comms-master/adversarial_comms/test_metric/cover_distribution_1v9")
    args = parser.parse_args()

    fig_overview = plt.figure(figsize=[5, 5])
    ax = fig_overview.subplots(1, 1)

    df1 = pd.read_pickle(df1)
    df2 = pd.read_pickle(df2)
    df3 = pd.read_pickle(df3)
    df4 = pd.read_pickle(df4)
    # df5 = pd.read_pickle(df5)
    # df6 = pd.read_pickle(df6)
    # df7 = pd.read_pickle(df7)

    plot_agent_role(ax ,df1[(df1['comm'] == False) & (df1['agent'] > 0)], 'blue', linestyle='-')
    plot_agent_role(ax ,df2[(df2['comm'] == False) & (df2['agent'] > 0)], 'olive', linestyle='-')
    plot_agent_role(ax ,df3[(df3['comm'] == False) & (df3['agent'] > 0)], 'orange', linestyle='-')
    plot_agent_role(ax ,df4[(df4['comm'] == False) & (df4['agent'] > 0)], 'lime', linestyle='-')

    plot_agent_role_explore(ax ,df1[(df1['comm'] == False) & (df1['agent'] > 0)], 'blue', linestyle=':')
    plot_agent_role_explore(ax ,df2[(df2['comm'] == False) & (df2['agent'] > 0)], 'olive', linestyle=':')
    plot_agent_role_explore(ax ,df3[(df3['comm'] == False) & (df3['agent'] > 0)], 'orange', linestyle=':')
    plot_agent_role_explore(ax ,df4[(df4['comm'] == False) & (df4['agent'] > 0)], 'lime', linestyle=':')
    # plot_agent_role(ax ,df5[(df5['comm'] == False) & (df5['agent'] > 0)], 'navy', linestyle='-')
    # plot_agent_role(ax ,df6[(df6['comm'] == False) & (df6['agent'] > 0)], 'slateblue', linestyle='-')
    # plot_agent_role(ax ,df7[(df7['comm'] == False) & (df7['agent'] > 0)], 'orchid', linestyle='-')


    line_1, = ax.plot([], [], color='blue', linestyle='-')
    line_2, = ax.plot([], [], color='olive', linestyle='-')
    line_3, = ax.plot([], [], color='orange', linestyle='-')
    line_4, = ax.plot([], [], color='lime', linestyle='-')

    line_5, = ax.plot([], [], color='blue', linestyle=':')
    line_6, = ax.plot([], [], color='olive', linestyle=':')
    line_7, = ax.plot([], [], color='orange', linestyle=':')
    line_8, = ax.plot([], [], color='lime', linestyle=':')
    # line_5, = ax.plot([], [], color='navy', linestyle='-')
    # line_6, = ax.plot([], [], color='slateblue', linestyle='-')
    # line_7, = ax.plot([], [], color='orchid', linestyle='-')
    
    ax.set_ylabel("role distribution %")
    ax.set_ylim(0, 1)
    ax.set_xlabel("step")
    timestep = 50
    ax.set_xlim(0, timestep)

    ax.legend(handles=[line_1,line_2,line_3,line_4,line_5,line_6,line_7,line_8], labels=['Easy-Cover',
                                                                                         'Medium-Cover',
                                                                                         'Hard-Cover',
                                                                                         'SuperHard-Cover',
                                                                                         'Easy-Explore',
                                                                                         'Medium-Explore',
                                                                                         'Hard-Explore',
                                                                                         'SuperHard-Explore',], loc='best')

    ax.margins(x=0, y=0)
    ax.grid()
    fig_overview.tight_layout()
    if args.out_file is not None:
        fig_overview.savefig(args.out_file, dpi=300)

    plt.show()

if __name__ == '__main__':

    varify_explored = True
    varify_coverage = False

    # varify_explored = False
    # varify_coverage = True
    
    # interest_0.08
    df_01 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio_robots_8/eval_coop-interst_0.08-robots_8-0.1explore_0.9cover-checkpoint-.pkl"
    # df_01 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio_robots_8/eval_coop-interst_0.08-robots_8-0.2explore_0.8cover-checkpoint-.pkl"
    # df_01 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio_robots_8/eval_coop-interst_0.08-robots_8-0.3explore_0.7cover-checkpoint-.pkl"
    # df_01 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio_robots_8/eval_coop-interst_0.08-robots_8-0.4explore_0.6cover-checkpoint-.pkl"
    # timestep = 50
    # max_cov = 0.08*50*50

    # interest_0.16 with robots 4
    df_02 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-0.1explore_0.9cover-checkpoint-.pkl"
    # df_02 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-0.2explore_0.8cover-checkpoint-.pkl"
    # df_02 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-0.3explore_0.7cover-checkpoint-.pkl"
    # df_02 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-0.4explore_0.6cover-checkpoint-.pkl"
    # timestep = 50
    # max_cov = 0.16*25*25

    # # interest_0.24
    df_03 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.24-0.1explore_0.9cover-checkpoint-.pkl"
    # df_03 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.24-0.2explore_0.8cover-checkpoint-.pkl"
    # df_03 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.24-0.3explore_0.7cover-checkpoint-.pkl"
    # df_03 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.24-0.4explore_0.6cover-checkpoint-.pkl"
    # timestep = 50
    # max_cov = 0.24*25*25

    # interest_0.56
    df_04 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.56-0.1explore_0.9cover-checkpoint-.pkl"
    # df_04 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.56-0.2explore_0.8cover-checkpoint-.pkl"
    # df_04 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.56-0.3explore_0.7cover-checkpoint-.pkl"
    # df_04 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.56-0.4explore_0.6cover-checkpoint-.pkl"
    # timestep = 50
    # max_cov = 0.56*25*25

    # plot(df_01,df_02,df_03,df_04,timestep,varify_explored,varify_coverage,max_cov) # 无自私机器人评估
    plot_role(df_04,df_03,df_02,df_01) # 无自私机器人评估
