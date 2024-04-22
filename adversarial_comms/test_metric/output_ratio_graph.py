import argparse
from jsonschema import draft201909_format_checker 
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pdb

GNN_data = {'coverd_ratio_each_step':[0,0.0020,0.0061,0.0082,0.01685333,0.025,0.03368,0.04184,0.0504,0.05938667,
                                      0.06866667,0.07753333,0.08654667,0.09529333,0.10377333,0.11309333,0.12214667,
                                      0.13108,0.1394,0.16438667,0.17304,0.18152,0.19,0.19766667,0.20548,0.21296,
                                      0.22036,0.22794667,0.23522667,0.24224,0.24969333,0.25661333,0.26348,0.27018667,
                                      0.27673333,0.28281333,0.28901333,0.29534667,0.30141333,0.30765333,0.31329333,
                                      0.31865333,0.32381333,0.32914667,0.3342,0.33914667,0.34397333,0.34869333,0.35334667,0.35813333],
            'explored_ratio_each_step':[0,0.15678933,0.18400533,0.211472,0.23844267,0.264944,0.29112533,0.31647467,
                                        0.34138133,0.36569067,0.389328,0.41258133,0.435312,0.45749333,0.47896,
                                        0.499632,0.519472,0.53848,0.55682133,0.57433067,0.59104533,0.60705067,
                                        0.622624,0.637472,0.65139733,0.66413867,0.676272,0.68736533,0.69793067,
                                        0.70809067,0.71754667,0.726096,0.73426667,0.742048,0.749152,0.755808,
                                        0.76174933,0.767296,0.77254933,0.777392,0.78200533,0.78632,0.79045333,
                                        0.79444267,0.79818667,0.80173867,0.805216,0.80863467,0.81184,0.814656]}

H2GNN_data = {'coverd_ratio_each_step':[0,0.01156,0.02365333,0.03678667,0.05017333,0.06350667,0.07670667,0.08982667,
                                        0.10304,0.11604,0.12922667,0.14209333,0.15470667,0.16792,0.18069333,0.19362667,
                                        0.2056,0.21769333,0.23021333,0.24236,0.25410667,0.26617333,0.27796,0.28952,0.3014,
                                        0.31274667,0.324,0.33521333,0.34636,0.35693333,0.36754667,0.378,0.38808,0.39882667,
                                        0.40889333,0.41873333,0.42838667,0.43801333,0.44750667,0.45702667,0.46589333,0.47484,
                                        0.48358667,0.49237333,0.50092,0.5092,0.51798667,0.52582667,0.53418667,0.54189333],
            'explored_ratio_each_step':[0,0.155632,0.18165333,0.20745067,0.232672,0.257184,0.281488,0.30501867,0.32829867,
                                        0.35111467,0.37364267,0.39538667,0.416352,0.43707733,0.456976,0.47656533,0.49572267,
                                        0.51443733,0.5328,0.550848,0.56822933,0.58482133,0.60021867,0.615088,0.62872,0.64154667,
                                        0.653568,0.66492267,0.67551467,0.68546667,0.69449067,0.70346133,0.71234133,0.720832,
                                        0.72904533,0.73676267,0.74416,0.75130667,0.75805333,0.764432,0.770784,0.77683733,0.78234667,
                                        0.78723733,0.79184,0.79643733,0.80082133,0.80519467,0.80935467,0.813424]}

VRPC_data = {'coverd_ratio_each_step' : [0,0.01466667,0.03066667,0.045,0.05966667,0.07633333,0.092,0.10733333,0.123,0.138,0.153,
                                         0.16766667,0.18233333,0.19733333,0.206,0.22033333,0.234,0.24833333,0.26,0.275,0.288,0.3,
                                         0.31266667,0.324,0.33633333,0.347,0.35566667,0.366,0.37466667,0.38333333,0.39233333,0.404,
                                         0.41233333,0.42133333,0.43,0.441,0.45033333,0.459,0.46666667,0.47433333,0.48333333,0.491,
                                         0.49766667,0.50666667,0.51333333,0.52133333,0.529,0.53633333,0.544,0.551], 
            'explored_ratio_each_step': [0,0.16653333,0.18613333,0.20786667,0.2268,0.246,0.2664,0.28333333,0.30373333,0.3212,
                                         0.33906667,0.3524,0.364,0.37546667,0.3888,0.39933333,0.41133333,0.42226667,0.43413333,
                                         0.44613333,0.45706667,0.47,0.48346667,0.49466667,0.506,0.51493333,0.52466667,0.5352,
                                         0.5444,0.55533333,0.56426667,0.5744,0.58466667,0.59733333,0.60586667,0.6168,0.62653333,
                                         0.63413333,0.64213333,0.6516,0.66146667,0.668,0.67453333,0.684,0.6916,0.6984,0.70466667,
                                         0.71146667,0.7164,0.72066667]}

def plot_agent(ax, df, color, linestyle='-',linewidth=1):
    world_shape = df.attrs['env_config']['world_shape']
    d = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['explored_ratio'].apply("mean", 'step').groupby('trial').cumsum()).groupby('step')
    ax.plot(d.mean()*100, color=color, ls=linestyle, linewidth=linewidth)
    # ax.fill_between(np.arange(len(d.mean())), np.clip(d.mean()-d.std(), 0, None), d.mean()+d.std(), alpha=0.1, color=color)

def plot_agent_loudian(ax, df, color, linestyle='-',linewidth=2):
    world_shape = df.attrs['env_config']['world_shape']
    d_ = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['covered_ratio'].apply('mean', 'step').groupby('step').apply('mean','trial')).groupby('step')
    ax.plot(d_.mean(), color=color, ls=linestyle, linewidth=linewidth)

def plot_agent_role(ax, df, color, linestyle='-',linewidth=2):
    world_shape = df.attrs['env_config']['world_shape']
    d_ = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['role_cover_ratio'].apply('mean', 'step').groupby('step').apply('mean','trial')).groupby('step')
    ax.plot(d_.mean(), color=color, ls=linestyle, linewidth=linewidth)

def plot_agent_coverage(ax, df, color, step_aggregation='sum', linestyle='-',linewidth=1):
    world_shape = df.attrs['env_config']['world_shape']
    max_cov = 0.24 * 25 *25
    d = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['reward_coverage'].apply(step_aggregation, 'step').groupby('trial').cumsum()/max_cov*100).groupby('step')
    ax.plot(d.mean(), color=color, ls=linestyle, linewidth=linewidth)
    # ax.fill_between(np.arange(len(d.mean())), np.clip(d.mean()-d.std(), 0, None), d.mean()+d.std(), alpha=0.05, color=color)

def plot_agent__(ax, df, color, linestyle='-',linewidth=1):
    world_shape = df.attrs['env_config']['world_shape']
    d_ = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['loudian_ratio'].apply('mean', 'step').groupby('step').apply('mean','trial')).groupby('step')
    ax.plot(d_.mean(), color=color, ls=linestyle, linewidth=linewidth)

def plot(df,df_01,df_02,df_03,df_end2end,df_greedy,df_random,timestep,varify_explored):
    df1 = df_end2end
    df2 = df_greedy
    df3 = df_random
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_file", default="/home/zln/repos/adversarial_comms-master/adversarial_comms/test_metric/test_covered_ratio")
    args = parser.parse_args()

    fig_overview = plt.figure(figsize=[5, 5])
    ax = fig_overview.subplots(1, 1)

    df = pd.read_pickle(df)
    df_01 = pd.read_pickle(df_01)
    df_02 = pd.read_pickle(df_02)
    df_03 = pd.read_pickle(df_03)
    df1 = pd.read_pickle(df1)
    df2 = pd.read_pickle(df2)
    df3 = pd.read_pickle(df3)

    if varify_explored:
        plot_agent(ax ,df[(df['comm'] == False) & (df['agent'] > 0)], 'r', linestyle='-')
        plot_agent(ax ,df_01[(df_01['comm'] == False) & (df_01['agent'] > 0)], 'black', linestyle='-')
        plot_agent(ax ,df_02[(df_02['comm'] == False) & (df_02['agent'] > 0)], 'olive', linestyle='-')
        plot_agent(ax ,df_03[(df_03['comm'] == False) & (df_03['agent'] > 0)], 'orange', linestyle='-')
        # plot_agent(ax ,df_04[(df_04['comm'] == False) & (df_04['agent'] > 0)], 'lime', linestyle='-')
        # plot_agent(ax ,df_06[(df_06['comm'] == False) & (df_06['agent'] > 0)], 'navy', linestyle='-')
        # plot_agent(ax ,df_07[(df_07['comm'] == False) & (df_07['agent'] > 0)], 'slateblue', linestyle='-')
        # plot_agent(ax ,df_08[(df_08['comm'] == False) & (df_08['agent'] > 0)], 'orchid', linestyle='-')
        # plot_agent(ax ,df_09[(df_04['comm'] == False) & (df_09['agent'] > 0)], 'purple', linestyle='-')
        plot_agent(ax ,df1[(df1['comm'] == False) & (df1['agent'] > 0)], 'cyan', linestyle='-')
        # plot_agent(ax ,df1_fensan[(df1_fensan['comm'] == False) & (df1_fensan['agent'] > 0)], 'deepskyblue', linestyle='-')
        # plot_agent(ax ,df1_jizhong[(df1_jizhong['comm'] == False) & (df1_jizhong['agent'] > 0)], 'deepskyblue', linestyle='-')
        plot_agent(ax ,df2[(df2['comm'] == False) & (df2['agent'] > 0)], 'g', linestyle='-')
        plot_agent(ax ,df3[(df3['agent'] > 0)], 'b', linestyle='-')
    else:
        
        # plot_agent_coverage(ax ,df_04[(df_04['comm'] == False) & (df_04['agent'] > 0)], 'lime', linestyle='-')
        # plot_agent_coverage(ax ,df_06[(df_06['comm'] == False) & (df_06['agent'] > 0)], 'navy', linestyle='-')
        # plot_agent_coverage(ax ,df_07[(df_07['comm'] == False) & (df_07['agent'] > 0)], 'slateblue', linestyle='-')
        # plot_agent_coverage(ax ,df_08[(df_08['comm'] == False) & (df_08['agent'] > 0)], 'orchid', linestyle='-')
        # plot_agent_coverage(ax ,df_09[(df_04['comm'] == False) & (df_09['agent'] > 0)], 'purple', linestyle='-')
        plot_agent_coverage(ax ,df1[(df1['comm'] == True) & (df1['agent'] > 0)], 'cyan', linestyle='-')
        plot_agent_coverage(ax ,df2[(df2['comm'] == True) & (df2['agent'] > 0)], 'g', linestyle='-')
        plot_agent_coverage(ax ,df3[(df3['comm'] == True) & (df3['agent'] > 0)], 'b', linestyle='-')
        plot_agent_coverage(ax ,df[(df['comm'] == True) & (df['agent'] > 0)], 'r', linestyle='-')
        # plot_agent_coverage(ax ,df_00[(df_00['comm'] == False) & (df_00['agent'] > 0)], 'black', linestyle='-')
        # # plot_agent(ax ,df_1[(df_1['comm'] == False) & (df_1['agent'] > 0)], 'black', linestyle='-')
        plot_agent_coverage(ax ,df_01[(df_01['comm'] == True) & (df_01['agent'] > 0)], 'black', linestyle='-')
        plot_agent_coverage(ax ,df_02[(df_02['comm'] == True) & (df_02['agent'] > 0)], 'olive', linestyle='-')
        plot_agent_coverage(ax ,df_03[(df_03['comm'] == True) & (df_03['agent'] > 0)], 'orange', linestyle='-')


    # line_00, = ax.plot([], [], color='black', linestyle='-')

    # line_04, = ax.plot([], [], color='lime', linestyle='-')
    # line_06, = ax.plot([], [], color='navy', linestyle='-')
    # line_07, = ax.plot([], [], color='slateblue', linestyle='-')
    # line_08, = ax.plot([], [], color='orchid', linestyle='-')
    # line_09, = ax.plot([], [], color='purple', linestyle='-')
    line1, = ax.plot([], [], color='cyan', linestyle='-')
    line2, = ax.plot([], [], color='g', linestyle='-')
    line3, = ax.plot([], [], color='b', linestyle='-')
    line, = ax.plot([], [], color='r', linestyle='-')
    line_01, = ax.plot([], [], color='black', linestyle='-')
    line_02, = ax.plot([], [], color='olive', linestyle='-')
    line_03, = ax.plot([], [], color='orange', linestyle='-')


    line4, = ax.plot([], [], color='purple', linestyle='-',linewidth=1)
    line5, = ax.plot([], [], color='slateblue', linestyle='-',linewidth=1)
    line6, = ax.plot([], [], color='lime', linestyle='-',linewidth=1)
   

    line4.set_xdata(np.array([i for i in range(timestep)]))
    line5.set_xdata(np.array([i for i in range(timestep)]))
    line6.set_xdata(np.array([i for i in range(timestep)]))
    if varify_explored:
        line4.set_ydata(np.array(GNN_data['explored_ratio_each_step'])*100)
        line5.set_ydata(np.array(H2GNN_data['explored_ratio_each_step'])*100)
        line6.set_ydata(np.array(VRPC_data['explored_ratio_each_step'])*100)
    else:
        line4.set_ydata(np.array(GNN_data['coverd_ratio_each_step'])*100)
        line5.set_ydata(np.array(H2GNN_data['coverd_ratio_each_step'])*100)
        line6.set_ydata(np.array(VRPC_data['coverd_ratio_each_step'])*100)

    if varify_explored:
        ax.set_ylabel("exploration percentage %")
        ax.set_ylim(0, 100)
        ax.set_xlabel("step")
        ax.set_xlim(0, timestep)
    else:
        ax.set_ylabel("coverage percentage %")
        ax.set_ylim(0, 65)
        ax.set_xlabel("step")
        ax.set_xlim(0, timestep)

    # ax.legend(handles=[line,line_1,line_01,line_02,line_03,line_04,line1,line2,line3], labels=['Role-A_1-B_0',
    #                                                     'Role-A_1-B_1',
    #                                                     'Role-A_0.1-B_0.9',
    #                                                     'Role-A_0.2-B_0.8',
    #                                                     'Role-A_0.3-B_0.7',
    #                                                     'Role-A_0.4-B_0.6',
    #                                                     'End2end',"Random","Greedy"], loc='best')

    ax.legend(handles=[line2,line3,line6,line4,line5,line1,line_01,line_02,line_03,line], labels=["Greedy","Random",'VRPC','GNN','H2GNN','ECC',
                                                                                'Ours-E_0.1-C_0.9',
                                                                                'Ours-E_0.2-C_0.8', 
                                                                                'Ours-E_0.3-C_0.7',
                                                                                'Ours-E_0.4-C_0.6'
                                                                                ], loc='best')

    ax.margins(x=0, y=0)
    ax.grid()
    # fig_overview.canvas.draw()
    # fig_overview.canvas.flush_events()
    fig_overview.tight_layout()
    if args.out_file is not None:
        fig_overview.savefig(args.out_file, dpi=300)

    plt.show()
    
def plot_role(df1,df3,df7):
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--out_file", default="/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/output_graph/different_inte_46")
    args = parser.parse_args()

    fig_overview = plt.figure(figsize=[5, 5])
    ax = fig_overview.subplots(1, 1)

    df1 = pd.read_pickle(df1)
    # df2 = pd.read_pickle(df2)
    df3 = pd.read_pickle(df3)
    # df4 = pd.read_pickle(df4)
    # df5 = pd.read_pickle(df5)
    # df6 = pd.read_pickle(df6)
    df7 = pd.read_pickle(df7)

    plot_agent_role(ax ,df1[(df1['comm'] == False) & (df1['agent'] > 0)], 'yellow', linestyle='-')
    # plot_agent_role(ax ,df2[(df2['comm'] == False) & (df2['agent'] > 0)], 'olive', linestyle='-')
    plot_agent_role(ax ,df3[(df3['comm'] == False) & (df3['agent'] > 0)], 'orange', linestyle='-')
    # plot_agent_role(ax ,df4[(df4['comm'] == False) & (df4['agent'] > 0)], 'lime', linestyle='-')
    # plot_agent_role(ax ,df5[(df5['comm'] == False) & (df5['agent'] > 0)], 'navy', linestyle='-')
    # plot_agent_role(ax ,df6[(df6['comm'] == False) & (df6['agent'] > 0)], 'slateblue', linestyle='-')
    plot_agent_role(ax ,df7[(df7['comm'] == False) & (df7['agent'] > 0)], 'orchid', linestyle='-')


    line_1, = ax.plot([], [], color='yellow', linestyle='-')
    # line_2, = ax.plot([], [], color='olive', linestyle='-')
    line_3, = ax.plot([], [], color='orange', linestyle='-')
    # line_4, = ax.plot([], [], color='lime', linestyle='-')
    # line_5, = ax.plot([], [], color='navy', linestyle='-')
    # line_6, = ax.plot([], [], color='slateblue', linestyle='-')
    line_7, = ax.plot([], [], color='orchid', linestyle='-')
    
    ax.set_ylabel("role cover selected ratio %")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Episode time steps in Role-A_0.4-B_0.6")
    ax.set_xlim(0, timestep)

    ax.legend(handles=[line_1,line_3,line_7], labels=['interest_point_ratio_0.08', 
                                                      'interest_point_ratio_0.24',
                                                      'interest_point_ratio_0.56'], loc='best')

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
    # df = "/home/zln/adv_results/search_rescue/0725_interest_100/train_only_explore/t_50/eval_coop-checkpoint-.pkl"  #train_only_loudian
    # df_1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_loudian_1v1_0728/eval_coop-checkpoint-.pkl"  #train_both_explore_loudian_1v1
    # # df_01 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_loudian/eval_coop-0.1explre-checkpoint-.pkl"  #train_both_explore_loudian_0.1v0.9
    # # df_02 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_loudian/eval_coop-0.2explre-checkpoint-.pkl"  #train_both_explore_loudian_0.2v0.8
    # # df_03 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_loudian/eval_coop-0.3explre-checkpoint-.pkl"  #train_both_explore_loudian_0.3v0.7
    # # df_04 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_loudian/eval_coop-0.4explre-checkpoint-.pkl"  #train_both_explore_loudian_0.4v0.6
    # df_01 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_loudian/eval_coop-0.6explre-checkpoint-.pkl"  #train_both_explore_loudian_0.1v0.9
    # df_02 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_loudian/eval_coop-0.7explre-checkpoint-.pkl"  #train_both_explore_loudian_0.2v0.8
    # df_03 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_loudian/eval_coop-0.8explre-checkpoint-.pkl"  #train_both_explore_loudian_0.3v0.7
    # df_04 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_loudian/eval_coop-0.9explre-checkpoint-.pkl"  #train_both_explore_loudian_0.4v0.6
    # df1 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/end2end_eval_coop-checkpoint-.pkl" #end2end
    # df2 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/random/eval_coop-checkpoint-.pkl" #random
    # df3 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/greedy/greedy_eval_coop-checkpoint-.pkl" #greedy
    # timestep=50

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

    # # interest_100_t_50_train_explore_cover
    # df = "/home/zln/adv_results/search_rescue/0725_interest_100/train_only_explore/t_50/eval_coop-checkpoint-.pkl"  #train_only_explore
    # df_00 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.0explore_1.0cover-checkpoint-.pkl"
    # df_01 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.1explore_0.9cover-checkpoint-.pkl"  #train_explore_cover_0.1v0.9
    # df_02 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.2explore_0.8cover-checkpoint-.pkl"  #train__explore_cover_0.2v0.8
    # df_03 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.3explore_0.7cover-checkpoint-.pkl"  #train_explore_cover_0.3v0.7
    # df_04 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.4explore_0.6cover-checkpoint-.pkl"  #train_explore_cover_0.4v0.6
    # df_06 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.6explore_0.4cover-checkpoint-.pkl"  #train_explore_cover_0.6v0.4
    # df_07 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.7explore_0.3cover-checkpoint-.pkl"  #train_explore_cover_0.7v0.3
    # df_08 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.8explore_0.2cover-checkpoint-.pkl"  #train_explore_cover_0.8v0.2
    # df_09 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.9explore_0.1cover-checkpoint-.pkl"  #train_explore_cover_0.9v0.1
    # df1 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/end2end_eval_coop-checkpoint-.pkl" #end2end
    # df1_fensan = "/home/zln/adv_results/search_rescue/end2end/end2end_fensan_eval_coop-checkpoint-.pkl" #end2end_fensan
    # df1_jizhong = "/home/zln/adv_results/search_rescue/end2end/end2end_jizhong_eval_coop-checkpoint-.pkl" #end2end_jizhong
    # df2 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/random/eval_coop-checkpoint-.pkl" #random
    # # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/greedy/greedy_eval_coop-checkpoint-.pkl" #greedy
    # # df3 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/greedy/greedy_eval_coop-checkpoint-.pkl" #greedy
    # # df3 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/greedy/eval_coop-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/greedy/eval_coop-checkpoint-.pkl"
    # timestep=50

    # # interest_100_t_50_train_explore_cover_re_2
    # df = "/home/zln/adv_results/search_rescue/0725_interest_100/train_only_explore/t_50/eval_coop-checkpoint-.pkl"  #train_only_explore
    # df_00 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.0explore_1.0cover-checkpoint-.pkl"
    # df_01 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.1explore_0.9cover-checkpoint-.pkl"  #train_explore_cover_0.1v0.9
    # df_02 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.2explore_0.8cover-checkpoint-.pkl"  #train__explore_cover_0.2v0.8
    # df_03 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.3explore_0.7cover-checkpoint-.pkl"  #train_explore_cover_0.3v0.7
    # df_04 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.4explore_0.6cover-checkpoint-.pkl"  #train_explore_cover_0.4v0.6
    # df_06 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.6explore_0.4cover-checkpoint-.pkl"  #train_explore_cover_0.6v0.4
    # df_07 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.7explore_0.3cover-checkpoint-.pkl"  #train_explore_cover_0.7v0.3
    # df_08 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.8explore_0.2cover-checkpoint-.pkl"  #train_explore_cover_0.8v0.2
    # df_09 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/eval_coop-0.9explore_0.1cover-checkpoint-.pkl"  #train_explore_cover_0.9v0.1
    # df1 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/end2end_eval_coop-checkpoint-.pkl" #end2end
    # df1_fensan = "/home/zln/adv_results/search_rescue/end2end/end2end_fensan_eval_coop-checkpoint-.pkl" #end2end_fensan
    # df1_jizhong = "/home/zln/adv_results/search_rescue/end2end/end2end_jizhong_eval_coop-checkpoint-.pkl" #end2end_jizhong
    # df2 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/random/eval_coop-checkpoint-.pkl" #random
    # # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/greedy/greedy_eval_coop-checkpoint-.pkl" #greedy
    # # df3 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/greedy/greedy_eval_coop-checkpoint-.pkl" #greedy
    # # df3 = "/home/zln/adv_results/search_rescue/0721_interest_100/t_50/greedy/eval_coop-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/greedy/eval_coop-checkpoint-.pkl"
    # timestep=50


    
    # df_role_04 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-0.4explore_0.6cover-checkpoint-.pkl"
    # df_role_03 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-0.3explore_0.7cover-checkpoint-.pkl"
    # df_role_02 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-0.2explore_0.8cover-checkpoint-.pkl"
    # df_role_01 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-0.1explore_0.9cover-checkpoint-.pkl"

    # # different interesting number in train_explore_cover_0.1v0.9
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.08-0.1explore_0.9cover-checkpoint-.pkl"
    # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-0.1explore_0.9cover-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.24-0.1explore_0.9cover-checkpoint-.pkl"
    # df4 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.32-0.1explore_0.9cover-checkpoint-.pkl"

    # # different interesting number in train_explore_cover_0.2v0.8
    # df1 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.08-0.4explore_0.6cover-checkpoint-.pkl"
    # # df2 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-0.1explore_0.9cover-checkpoint-.pkl"
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.24-0.4explore_0.6cover-checkpoint-.pkl"
    # # df4 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.32-0.1explore_0.9cover-checkpoint-.pkl"
    # # df5 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.4-0.1explore_0.9cover-checkpoint-.pkl"
    # # df6 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.48-0.1explore_0.9cover-checkpoint-.pkl"
    # df7 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2/add_juese_ratio/eval_coop-interst_0.56-0.4explore_0.6cover-checkpoint-.pkl"

    # plot_role(df1,df3,df7)

    timestep = 50
    varify_explored = False

    df = '/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/robots_4/eval_coop-interst_0.24-coverable_0.6-robots_4-0.4explore_0.6cover-checkpoint-.pkl'
    df_01 = '/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/all_role_pkl/eval_coop-interst_0.24-coverable_0.6-robots_4-0.1explore_0.9cover-checkpoint-.pkl'
    df_02 = '/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/all_role_pkl/eval_coop-interst_0.24-coverable_0.6-robots_4-0.2explore_0.8cover-checkpoint-.pkl'
    df_03 = '/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/all_role_pkl/eval_coop-interst_0.24-coverable_0.6-robots_4-0.3explore_0.7cover-checkpoint-.pkl'
    df_end2end = '/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/end2end/end2end-robots_4-interst_0.24-coverable_0.6-eval_coop-checkpoint-.pkl'
    df_greedy = '/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/greedy/greedy-robots_4-interst_0.24-coverable_0.6-eval_coop-checkpoint-.pkl'
    df_random = '/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/random/Random-interst_0.24-coverable_0.6-robots_4-0.4explore_0.6cover-checkpoint-.pkl'

    plot(df,df_01,df_02,df_03,df_end2end,df_greedy,df_random,timestep,varify_explored) # 无自私机器人评估
