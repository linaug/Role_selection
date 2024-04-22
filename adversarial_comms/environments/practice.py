from asyncio import as_completed
import random
import pandas as pd
import pickle 

def obstacle():
    import random
    import numpy as np
    Y=X=24
    pos = np.array([random.randint(0, c) for c in [Y, X]])
    obstacles = []
    num_obstacle = 10
    while len(obstacles) < num_obstacle:
        obs_pos = np.array([random.randint(0, c) for c in [Y, X]])
        if obs_pos[0] != pos[0] and obs_pos[1]!= pos[1]:
            obstacles.append(obs_pos)

    print(obstacles)

def circle(radius,x_center,y_center):
    import numpy as np
    import random
    from itertools import product
    def points_in_circle(radius_):
        for x, y in product(range(int(radius_) + 1), repeat=2):
            # print(x,y)
            if x**2 + y**2 <= radius_**2:
                yield from set(((x, y), (x, -y), (-x, y), (-x, -y),))

    Y=X=24
    pos = np.array([random.randint(0, c) for c in [Y, X]])
    print('pos:',pos)
    circles = []

    points = list(points_in_circle(radius))

    for circle in list(points):
        circles_ = [pos[0]+circle[0],pos[1]+circle[1]]
        circles.append(circles_)
        print("point:",circle)

    print('circles:',circles)

class AvgPooling(object):
    def __init__(self, shape, ksize=3, stride=1):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channels = shape[-1]
        self.integral = np.zeros(shape)
        self.index = np.zeros(shape)
        
    def backward(self, eta):
        next_eta = np.repeat(eta, self.stride, axis=1)
        next_eta = np.repeat(next_eta, self.stride, axis=2)
        next_eta = next_eta*self.index
        return next_eta/(self.ksize*self.ksize)

    def forward(self, x):
        out = np.zeros([x.shape[0], x.shape[1] / self.stride, x.shape[2] / self.stride, self.output_channels])

        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        out[b, i / self.stride, j / self.stride, c] = np.mean(
                            x[b, i:i + self.ksize, j:j + self.ksize, c])
                        
        return out

def exploration_froniter():
    from random import random
    import numpy as np
    import torch
    from torch import nn
    # img = torch.zeros(24*24).reshape(1,1,24,24)
    img = np.ones((8,8))
    pos =  [4,5]
    pos_1 = [3,1]
    for i in range(-1,2):
        for j in range(-1,2):
            img[pos[0]+i,pos[1]+j] = 0
            img[pos_1[0]+i,pos_1[1]+j] = 0

    # AvgPooling_ = AvgPooling(img.shape)
    # img_pool_np = AvgPooling_.forward(img.reshape(1,8,8))

    img = torch.from_numpy(img).reshape(1,1,8,8)
    pool = nn.AvgPool2d(3,stride=1,padding=1,ceil_mode=True)
    img_pool = pool(img) # 平均池化
    # bb = torch.round(img_pool)  # 大于0.5的置1
    cc = torch.where(img<1,1,0) & torch.where(((0<img_pool)&(img_pool<1)),1,0) # 取探索区域与池化后异常值的交集


    # print('img is \n {} \n img_ is \n {} \n {} \n cc is \n {}'.format(img,img_pool,cc))

    exploration = img

    obstacle = np.arange(8*8).reshape(8,8)

    obstacle_exploration = np.where(exploration==0,obstacle,0)

    print('exploration is \n {} \n obstacle is \n {} \n obstacle_exploration is \n {}'.format(exploration,obstacle,obstacle_exploration))



    shape=[24,24]+[2+5]
    print('shape:',shape)

def colormap():
    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import random
    import numpy as np
    import matplotlib.colors as mcolors

    x= np.linspace(1, 12, 24, endpoint=True)
    y=x/x

    fig = plt.figure()
    ax= plt.axes()
    ax.set_ylim(0.8, 1.2)

    plt.scatter(x, y*1.05,s=120, marker='s',c=x, cmap='Greys')
    plt.scatter(x, y*0.95,s=120, marker='s',c=x, cmap='magma')

    colors = ['Greys']
    cmap = mcolors.ListedColormap(colors)
    print("cm.Set1.colors:",cmap)

    plt.savefig("/home/zln/repos/adversarial_comms-master/adversarial_comms/environments/colormap.jpg", bbox_inches='tight')
    plt.show()

    hsv = np.ones((2, 3))
    hsv[..., 0] = np.linspace(160/360, 250/360, self.cfg['n_agents'][1] + 1)[:-1]

    self.teams_agents_color = {
    0: [(1, 0, 0)],
    1: colors.hsv_to_rgb(hsv)}

def seed(seed):
    from gym.utils import seeding, EzPickle

 
    agent_random_state, seed_agents = seeding.np_random(seed)
    world_random_state, seed_world = seeding.np_random(seed)
    return [seed_agents, seed_world]


def average_reward():
    import numpy as np
    import pandas as pd

    df = pd.read_pickle('/home/zln/adv_results/search_rescue/eval_coop-checkpoint-.pkl')
    df_1 = df
    len = 50*500*5  # 345*10
    reward_avg = None
    reward_avg_ = None
    for comm_ in range(2):
        reward_avg = df_1.iloc[int(comm_*len):int((comm_+1)*len)]['reward']
        # np.savetxt('/home/zln/adv_results/coop_0301_02/test_file.txt', reward_avg, fmt = "%d")
        reward_avg_ = reward_avg.to_numpy().astype(np.float64)
        reward_avg_ = np.reshape(reward_avg_, (500, 50, 5))
        reward_avg_ = np.sum(reward_avg_,axis=2)   
        reward_avg_ = np.sum(reward_avg_,axis=1)    
        reward_avg_ = np.mean(reward_avg_, axis=0)
        print("average_reward:",reward_avg_)
        # np.savetxt(Path(args.out_path)/"{}.txt".format(comm_), reward_avg_, fmt = "%.18f")
    
    # df1 = df_1[(df_1['comm']==True)]
    # d = (df1.sort_values(['trial', 'step']).groupby(['trial', 'step'])['reward'].groupby('trial').cumsum())
    # reward_avg_ = d.means()
    # print("average_reward:",reward_avg_)
    # df.sort_values是整理给出的栏，如上就是整理trail和step栏.

def loadtofile():
    import sys
    import torch
    import numpy as np
    mylog = open('/home/zln/ray_results/recode.log',mode='a',encoding='utf-8')
    for i in range(10):
        print("adcccccc",file=mylog)
    a = torch.tensor([1,2,3,4])
    b = torch.tensor([2,2,3,2])
    c = a - b
    
    print("c",c)
    print(c < 1)
    print(((c<1)==True).sum().item())
    mylog.close()

def output_vedio():

    import matplotlib.pyplot as plt
    import cv2 as cv
    import matplotlib.animation as animation
    import os
    import glob

    save_file = '/home/zln/adv_results/search_rescue/0717_model_add_loudian/150_timestep/'
    max_episode_len = 150
    fig = plt.figure()
    ims = []

    for ii in range(max_episode_len):
        # 用opencv读取图片
        img = cv.imread(save_file+str(ii)+'.png')
        (r, g, b) = cv.split(img) 
        img = cv.merge([b,g,r])  
        im = plt.imshow(img, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True, repeat_delay=False)
    ani.save(save_file+"movie0.mp4")
    plt.show()

    filelist = glob.glob(os.path.join(save_file, "*.png"))
    for f in filelist:
        os.remove(f)

def pandas_sers():
    import pandas as pd

    df1 = pd.DataFrame({'group': ['a', 'a', 'b', 'b'], 'values': [1, 1, 2, 2]})
    g1 = df1.groupby('group')
    g1_1st_column = g1['group'].apply(pd.DataFrame)
    print(type(g1_1st_column))

def pkltotxt():
    df = '/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover/Greedy_eval_coop-checkpoint-.pkl'
    df1 = pd.read_pickle(df)
    ft = open('test.txt','w')
    ft.write(df1)

def data():
    import numpy as np
    a = [i for i in range(50)]
    print(np.array(a)*10)

    GNN_data = {
        'coverd_ratio_each_step':[0,0.00808,0.01696,0.0252,0.03348,0.04192,0.05096,0.0612,0.07008,0.07828,
                                0.08816,0.0978,0.10636,0.11608,0.12584,0.1348,0.14364,0.15308,0.16172,
                                0.1698,0.17768,0.18584,0.1946,0.20276,0.21052,0.21832,0.22628,0.23388,
                                0.24104,0.24844,0.25508,0.2622,0.269,0.2746,0.28108,0.28708,0.29352,
                                0.29896,0.30504,0.31076,0.31684,0.32144,0.32572,0.33004,0.33452,0.3394,
                                0.34368,0.34712,0.35152,0.35568],
        'explored_ratio_each_step':[0.15709867,0.183536,0.209808,0.235632,0.26155733,0.28677867,
                                    0.31159467,0.33591467,0.35966933,0.383488,0.40723733,0.430048,
                                    0.45208533,0.47341333,0.494304,0.514592,0.53403733,0.552576,0.570528,
                                    0.58763733,0.60402133,0.61973867,0.63456533,0.648256,0.661248,0.673616,
                                    0.685216,0.69565333,0.70552,0.71466133,0.7232,0.73130667,0.738976,0.74589867,
                                    0.75248,0.75843733,0.76396267,0.76917867,0.77392,0.77834133,0.78231467,0.78618667,
                                    0.78992533,0.79330133,0.79647467,0.79952533,0.802464,0.80506133,0.80764267]}
    GNN_data = {'coverd_ratio_each_step':[0,0.00808,0.01696,0.0252,0.03348,0.04192,0.05096,0.0612,0.07008,0.07828,
                              0.08816,0.0978,0.10636,0.11608,0.12584,0.1348,0.14364,0.15308,0.16172,
                              0.1698,0.17768,0.18584,0.1946,0.20276,0.21052,0.21832,0.22628,0.23388,
                              0.24104,0.24844,0.25508,0.2622,0.269,0.2746,0.28108,0.28708,0.29352,
                              0.29896,0.30504,0.31076,0.31684,0.32144,0.32572,0.33004,0.33452,0.3394,
                              0.34368,0.34712,0.35152,0.35568],
            'explored_ratio_each_step':[0,0.15709867,0.183536,0.209808,0.235632,0.26155733,0.28677867,
                                        0.31159467,0.33591467,0.35966933,0.383488,0.40723733,0.430048,
                                        0.45208533,0.47341333,0.494304,0.514592,0.53403733,0.552576,0.570528,
                                        0.58763733,0.60402133,0.61973867,0.63456533,0.648256,0.661248,0.673616,
                                        0.685216,0.69565333,0.70552,0.71466133,0.7232,0.73130667,0.738976,0.74589867,
                                        0.75248,0.75843733,0.76396267,0.76917867,0.77392,0.77834133,0.78231467,0.78618667,
                                        0.78992533,0.79330133,0.79647467,0.79952533,0.802464,0.80506133,0.80764267]}

    H2GNN_data = {'coverd_ratio_each_step':[0,0.01152,0.02384,0.0364,0.05068,0.06692,0.08228,0.09696,0.1118,0.12768,
                                        0.14412,0.1606,0.176,0.1912,0.20708,0.22268,0.23792,0.2528,0.26752,
                                        0.28168,0.29696,0.31136,0.32576,0.33996,0.35364,0.36644,0.3802,0.3922,
                                        0.40456,0.41612,0.42856,0.44104,0.4524,0.46384,0.47492,0.487,0.49944,
                                        0.51,0.51976,0.5298,0.53924,0.54888,0.55828,0.5688,0.57756,0.586,
                                        0.59412,0.60236,0.6102,0.6172],
            'explored_ratio_each_step':[0,0.15906667,0.187488,0.215392,0.24272,0.26950933,0.2956,0.32074133,0.34539733,
                                        0.370128,0.39420267,0.41722133,0.439376,0.46039467,0.48098667,0.50067733,0.51918933,
                                        0.53695467,0.55416,0.57058133,0.58596267,0.60029333,0.61425067,0.62747733,0.64024,
                                        0.65245333,0.66421333,0.6756,0.68635733,0.696512,0.70632533,0.71552,0.723936,0.73191467,
                                        0.73927467,0.74626667,0.75250133,0.75821333,0.763648,0.76901333,0.77407467,0.779008,
                                        0.783824,0.78825067,0.79241067,0.79616533,0.79966933,0.802816,0.80578667,0.80865067]}
    print('GNN covered_shape is{} and explored_shape is{}'.format(np.array(GNN_data['coverd_ratio_each_step']).shape,np.array(GNN_data['explored_ratio_each_step']).shape))
    print('H2GNN covered_shape is{} and explored_shape is{}'.format(np.array(H2GNN_data['coverd_ratio_each_step']).shape,np.array(H2GNN_data['explored_ratio_each_step']).shape))

def test():
    import argparse
from jsonschema import draft201909_format_checker
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_agent(ax, df, color, step_aggregation='sum', linestyle='-',linewidth=2):
    world_shape = df.attrs['env_config']['world_shape']
    max_cov = 0.24 * 25 *25
    d = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['reward_coverage'].apply(step_aggregation, 'step').groupby('trial').cumsum()/max_cov*100).groupby('step')
    # d = (df.sort_values(['trial', 'step']).groupby(['trial', 'step'])['explored_ratio'].apply("mean", 'step').groupby('trial').cumsum()).groupby('step')

    # df.sort_values是整理给出的栏，如上就是整理trail和step栏.
    ax.plot(d.mean(), color=color, ls=linestyle, linewidth=linewidth)
    ax.fill_between(np.arange(len(d.mean())), np.clip(d.mean()-d.std(), 0, None), d.mean()+d.std(), alpha=0.1, color=color)

def test():
    def plot(df,df1,df2,df3):

        fig_overview = plt.figure(figsize=[4, 4])
        ax = fig_overview.subplots(1, 1)

        # df = pd.read_pickle(args.data)
        # df1 = pd.read_pickle(args.data1)
        # df2 = pd.read_pickle(args.data2)
        # df3 = pd.read_pickle(args.data3)
        df = pd.read_pickle(df)
        df1 = pd.read_pickle(df1)
        df2 = pd.read_pickle(df2)
        df3 = pd.read_pickle(df3)
    
        plot_agent(ax, df[(df['comm'] == True) & (df['agent'] > 0)], 'r', step_aggregation='sum', linestyle='-')
        # plot_agent(ax, df[(df['comm'] == False) & (df['agent'] > 0)], 'r', step_aggregation='sum', linestyle=':')

        plot_agent(ax, df1[(df1['comm'] == True) & (df1['agent'] > 0)], 'y', step_aggregation='sum', linestyle='-')
        # plot_agent(ax, df1[(df1['comm'] == False) & (df1['agent'] > 0)], 'b', step_aggregation='sum', linestyle=':')

        plot_agent(ax, df2[(df2['comm'] == True) & (df2['agent'] > 0)], 'g', step_aggregation='sum', linestyle='-')        
        plot_agent(ax, df3[(df3['comm'] == True) & (df3['agent'] > 0)], 'b', step_aggregation='sum', linestyle='-')        

        line, = ax.plot([], [], color='r', linestyle='-')
        line1, = ax.plot([], [], color='y', linestyle='-')
        line2, = ax.plot([], [], color='g', linestyle='-')
        line3, = ax.plot([], [], color='b', linestyle='-')

        ax.set_ylabel("Coverage %")
        # ax.set_ylim(92, 100)  # fov测试
        ax.set_ylim(0, 100)
        ax.set_xlabel("Episode time steps")
        ax.set_xlim(0,50)
        ax.legend(handles=[line, line1, line2, line3], labels=['TSC','TSC-GM', 'ECC', 'Greedy'], loc='best')
        # ax.legend(handles=[line, line3], labels=['Fov w/ 8×8','Fov w/ 24×24'], loc= 4)

        ax.margins(x=0, y=0)
        ax.grid()
        fig_overview.tight_layout()

        fig_overview.savefig('/home/zln/repos/adversarial_comms-master/adversarial_comms/test_metric/practice.png', dpi=300)

        plt.show()

    df_01 = '/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/all_role_pkl/eval_coop-interst_0.24-coverable_0.6-robots_4-0.1explore_0.9cover-checkpoint-.pkl'
    df_02 = '/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/all_role_pkl/eval_coop-interst_0.24-coverable_0.6-robots_4-0.2explore_0.8cover-checkpoint-.pkl'
    df_03 = '/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/all_role_pkl/eval_coop-interst_0.24-coverable_0.6-robots_4-0.3explore_0.7cover-checkpoint-.pkl'
    # df3 = "/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/all_role_pkl/Random-interst_0.08-coverable_0.6-robots_4-0.3explore_0.7cover-checkpoint-.pkl"
    plot(df_01,df_02,df_02,df_02) # 无自私机器人评估

def length_data():
    GNN_data = {'coverd_ratio_each_step':[0.0082,0.01685333,0.025,0.03368,0.04184,0.0504,0.05938667,
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

    H2GNN_data = {'coverd_ratio_each_step':[0.01156,0.02365333,0.03678667,0.05017333,0.06350667,0.07670667,0.08982667,
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
    print("all")


if __name__ == '__main__':
    
    # seed_ = int(str(random.randint(0,9)))
    # seed_agents, seed_world = seed(seed_)
    # print('seed_input is {} \n seed_agents is {}, seed_world is {}'.format(seed_,seed_agents,seed_world))
    # # a = [1,2,3]
    # a_average = sum(a)/len(a)
    # print('a_average:',a_average)

    # average_reward()
    # loadtofile()

    # output_vedio()
    # pandas_sers()
    # pkltotxt()
    # data()
    # test()
    length_data()




