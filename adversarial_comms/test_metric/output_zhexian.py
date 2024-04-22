import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import gca
import pandas as pd
import queue
import os
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}

CONFIG_MAP = {
    'x-dim':[576,900,1600,2500,3600,4900],
    'greedy':[150,250,299,399,499,599],
    'ecc':[69,111,190,329,336,599],
    'ours_greedy':[46,144,167,399,499,599],
    'ours':[45,79,142,231,318,465],
    'img_path':"SIZE_MAP.png"
}

CONFIG_NUM_Robots = {
    'x-dim':[5,10,15,20,25],
    'greedy':[1599,800,600,500,400],
    'ecc':[1381,739,463,329,266],
    'ours_greedy':[645,699,499,399,299],
    'ours':[815,457,319,231,170],
    'img_path':"NUM_Robots.png"
}

def plot(config):      
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # plt.xlim(4, 26)                     #  要改
    # plt.ylim(0, 1600)                   #  要改
    plt.xlim(500, 5500)               #  要改
    plt.ylim(0, 650)                  #  要改
    plt.xlabel('number of grid size')      #  要改
    plt.ylabel('step')                  #  要改
    line1, = ax.plot([], [], color='#56B371', marker= 'p',linestyle='-',linewidth=2)
    line2, = ax.plot([], [], color='#00C3FE', marker= '^',linestyle='-',linewidth=2)
    line3, = ax.plot([], [], color='#FFC43D', marker= '*',linestyle='-',linewidth=2)
    line4, = ax.plot([], [], color='#D84648', marker= 's',linestyle='-',linewidth=2)
    # line5, = ax.plot([], [], color='#465258', linestyle='-')
    # line6, = ax.plot([], [], color='#700B97', linestyle='-')

    # set the width of dims
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    # ax.spines['bottom'].set_color('#8F8F8F')
    # ax.spines['left'].set_color('#8F8F8F')
    # ax.spines['top'].set_color('#8F8F8F')
    # ax.spines['right'].set_color('#8F8F8F')
    # 00C3FE
    line1.set_xdata(np.array(config['x-dim']))
    line1.set_ydata(np.array(config['greedy']))
    line2.set_xdata(np.array(config['x-dim']))
    line2.set_ydata(np.array(config['ecc']))
    line3.set_xdata(np.array(config['x-dim']))
    line3.set_ydata(np.array(config['ours_greedy']))
    line4.set_xdata(np.array(config['x-dim']))
    line4.set_ydata(np.array(config['ours']))
    # line5.set_xdata(np.array(config['x-dim']))
    # line5.set_ydata(np.array(config['greedy']))
    # line6.set_xdata(np.array(config['x-dim']))
    # line6.set_ydata(np.array(config['vrpc']))
    plt.legend(handles=[line1, line2, line3, line4], labels=['Greedy', 'ECC', 'TSC-GM','TSC'], loc='best')
    # map note
    plt.grid(axis='y')
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.savefig(config['img_path'], dpi=800)
    plt.show()

if __name__=='__main__':

    plot(CONFIG_MAP)


