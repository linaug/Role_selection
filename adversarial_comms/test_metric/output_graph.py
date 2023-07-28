import argparse
from jsonschema import draft201909_format_checker 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb

def plot_agent(ax, df, color, linestyle='-',linewidth=2):
    world_shape = df.attrs['env_config']['world_shape']
    d = df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['explored_ratio'].apply('mean', 'step').groupby('step').apply('mean','trial').groupby('step')
    d_ = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['loudian_ratio'].apply('mean', 'step').groupby('step').apply('mean','trial')).groupby('step')

    pdb.set_trace()
    ax.plot(d.mean(), color=color, ls=linestyle, linewidth=linewidth)
    # ax.fill_between(np.arange(len(d.mean())), np.clip(d.mean()-d.std(), 0, None), d.mean()+d.std(), alpha=0.1, color=color)

def plot_agent_loudian(ax, df, color, linestyle='-',linewidth=2):
    world_shape = df.attrs['env_config']['world_shape']
    d_ = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['loudian_ratio'].apply('mean', 'step').groupby('step').apply('mean','trial')).groupby('step')
    ax.plot(d_.mean(), color=color, ls=linestyle, linewidth=linewidth)
    # ax.fill_between(np.arange(len(d.mean())), np.clip(d.mean()-d.std(), 0, None), d.mean()+d.std(), alpha=0.1, color=color)

def plot_agent_coverage(ax, df, color, step_aggregation='sum', linestyle='-'):
    world_shape = df.attrs['env_config']['world_shape']
    max_cov = 0.16 * 25 *25
    d = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['reward_coverage'].apply(step_aggregation, 'step').groupby('trial').cumsum()/max_cov*100).groupby('step')
    # df.sort_values是整理给出的栏，如上就是整理trail和step栏.
    ax.plot(d.mean(), color=color, ls=linestyle)
    # ax.fill_between(np.arange(len(d.mean())), np.clip(d.mean()-d.std(), 0, None), d.mean()+d.std(), alpha=0.1, color=color)

def plot_agent__(ax, df, color, linestyle='-',linewidth=2):
    world_shape = df.attrs['env_config']['world_shape']
    d_ = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['loudian_ratio'].apply('mean', 'step').groupby('step').apply('mean','trial')).groupby('step')
    ax.plot(d_.mean(), color=color, ls=linestyle, linewidth=linewidth)
    # ax.fill_between(np.arange(len(d.mean())), np.clip(d.mean()-d.std(), 0, None), d.mean()+d.std(), alpha=0.1, color=color)


def plot(df,df1,df2,df3,timestep,varify_explored,varify_loudian,varify_coverage):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_file", default=None)
    args = parser.parse_args()

    fig_overview = plt.figure(figsize=[4, 4])
    ax = fig_overview.subplots(1, 1)

    df = pd.read_pickle(df)
    df1 = pd.read_pickle(df1)
    df2 = pd.read_pickle(df2)
    df3 = pd.read_pickle(df3)

    # varify_explored = True
    # varify_loudian = False
    if varify_explored:
        plot_agent(ax ,df[(df['comm'] == False) & (df['agent'] > 0)], 'r', linestyle='-')
        plot_agent(ax ,df1[(df1['comm'] == False) & (df1['agent'] > 0)], 'y', linestyle='-')
        plot_agent(ax ,df2[(df2['comm'] == False) & (df2['agent'] > 0)], 'g', linestyle='-')
        plot_agent(ax ,df3[(df3['agent'] > 0)], 'b', linestyle='-')
    elif varify_loudian:
        plot_agent_loudian(ax ,df[(df['comm'] == False) & (df['agent'] > 0)], 'r', linestyle=':')
        plot_agent_loudian(ax ,df1[(df1['comm'] == False) & (df1['agent'] > 0)], 'y', linestyle=':')
        plot_agent_loudian(ax ,df2[(df2['comm'] == False) & (df2['agent'] > 0)], 'g', linestyle=':')
        plot_agent_loudian(ax ,df3[(df3['agent'] > 0)], 'b', linestyle=':')
    elif varify_coverage:
        plot_agent_coverage(ax ,df[(df['comm'] == False) & (df['agent'] > 0)], 'r', linestyle='dashdot')
        plot_agent_coverage(ax ,df1[(df1['comm'] == False) & (df1['agent'] > 0)], 'y', linestyle='dashdot')
        plot_agent_coverage(ax ,df2[(df2['comm'] == False) & (df2['agent'] > 0)], 'g', linestyle='dashdot')
        plot_agent_coverage(ax ,df3[(df3['agent'] > 0)], 'b', linestyle='dashdot')


    line, = ax.plot([], [], color='r', linestyle='-')
    line1, = ax.plot([], [], color='y', linestyle='-')
    line2, = ax.plot([], [], color='g', linestyle='-')
    line3, = ax.plot([], [], color='b', linestyle='-')
    # # line4, = ax.plot([], [], color='grey', linestyle='-')
    if varify_explored:
        ax.set_ylabel("Explored ratio %")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Episode time steps")
        ax.set_xlim(0, timestep)
    if varify_coverage:
        ax.set_ylabel("Covered ratio %")
        ax.set_ylim(0, 100)
        ax.set_xlabel("Episode time steps")
        ax.set_xlim(0, timestep)
    elif varify_loudian:
        ax.set_ylabel("Forgetting ratio %")
        ax.set_ylim(0, 0.3)
        ax.set_xlabel("Episode time steps")
        ax.set_xlim(0, timestep)

    ax.legend(handles=[line,line1,line2,line3], labels=['Role-selection','End2end',"Random","Greedy"], loc='best')

    ax.margins(x=0, y=0)
    ax.grid()
    fig_overview.tight_layout()
    if args.out_file is not None:
        fig_overview.savefig(args.out_file, dpi=300)

    plt.show()

if __name__ == '__main__':
    # interset_150_t_150
    # df = "/home/zln/adv_results/search_rescue/0717_model_add_loudian/0720_/eval_coop-checkpoint-.pkl"
    # df1 = "/home/zln/adv_results/search_rescue/end2end/end2end-eval_coop-checkpoint-.pkl"

    # # interest_100_t_150
    # df = "/home/zln/adv_results/search_rescue/0721_100_interest/t_150/eval_coop-checkpoint-.pkl"
    # df1 = "/home/zln/adv_results/search_rescue/0721_100_interest/t_150/end2end_eval_coop-checkpoint-.pkl"

    # # interest_100_t_100
    # df = "/home/zln/adv_results/search_rescue/0721_100_interest/t_100/eval_coop-checkpoint-.pkl"
    # df1 = "/home/zln/adv_results/search_rescue/0721_100_interest/t_100/end2end_eval_coop-checkpoint-.pkl"

    # # # interest_100_t_50
    # df = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/eval_coop-checkpoint-.pkl"  #role-based
    # df1 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/end2end_eval_coop-checkpoint-.pkl" #end2end
    # df2 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/random/eval_coop-checkpoint-.pkl" #random
    # df3 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/greedy/greedy_eval_coop-checkpoint-.pkl" #greedy
    # timestep=50

    # # # interest_100_t_50_explore_reward
    # df = "/home/zln/adv_results/search_rescue/0725_interest_100/t_50/eval_coop-checkpoint-.pkl"  #role-based
    # df1 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/end2end_eval_coop-checkpoint-.pkl" #end2end
    # df2 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/random/eval_coop-checkpoint-.pkl" #random
    # df3 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/greedy/greedy_eval_coop-checkpoint-.pkl" #greedy
    # timestep=50

    # # interest_100_t_50_loudian_reward
    df = "/home/zln/adv_results/search_rescue/0725_interest_100/train_only_loudian/t_50/eval_coop-checkpoint-.pkl"  #role-based
    df1 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/end2end_eval_coop-checkpoint-.pkl" #end2end
    df2 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/random/eval_coop-checkpoint-.pkl" #random
    df3 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/greedy/greedy_eval_coop-checkpoint-.pkl" #greedy
    timestep=50

    # # # interest_50_t_150
    # df = "/home/zln/adv_results/search_rescue/0723_interest_50/t_150/eval_coop-checkpoint-.pkl"
    # df1 = "/home/zln/adv_results/search_rescue/0723_interest_50/t_150/end2end-eval_coop-checkpoint-.pkl"
    # timestep=150

    # # # interest_50_t_100
    # df = "/home/zln/adv_results/search_rescue/0723_interest_50/t_100/eval_coop-checkpoint-.pkl"
    # df1 = "/home/zln/adv_results/search_rescue/0723_interest_50/t_100/end2end-eval_coop-checkpoint-.pkl"
    # timestep=100

    # # interest_50_t_50
    # df = "/home/zln/adv_results/search_rescue/0723_interest_50/t_50/eval_coop-checkpoint-.pkl"
    # df1 = "/home/zln/adv_results/search_rescue/0723_interest_50/t_50/end2end-eval_coop-checkpoint-.pkl"
    # timestep=50

    # # # interest_50_t_150_robots_8
    # df = "/home/zln/adv_results/search_rescue/0721_interest_100/t_150/robots_8/eval_coop-checkpoint-.pkl"
    # df1 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_150/robots_8/end2end_eval_coop-checkpoint-.pkl"
    # timestep=150

    # # # interest_50_t_100_robots_8
    # df = "/home/zln/adv_results/search_rescue/0721_interest_100/t_100/robots_8/eval_coop-checkpoint-.pkl"
    # df1 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_100/robots_8/end2end_eval_coop-checkpoint-.pkl"
    # timestep=100

    # # interest_50_t_50_robots_8
    # df = "/home/zln/adv_results/search_rescue/0721_interest_100/t_100/robots_8/eval_coop-checkpoint-.pkl"
    # df1 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_100/robots_8/end2end_eval_coop-checkpoint-.pkl"
    # timestep=50

    varify_explored = True
    varify_loudian = False
    varify_coverage = False

    # varify_explored = True
    # varify_loudian = False
    # varify_coverage = False


    plot(df,df1,df2,df3,timestep,varify_explored,varify_loudian,varify_coverage) # 无自私机器人评估