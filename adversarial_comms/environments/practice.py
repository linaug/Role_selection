from asyncio import as_completed
import random


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
    pandas_sers()




