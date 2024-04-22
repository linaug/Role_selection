import csv  # 导入csv模块
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import gca
import pandas as pd
import queue
import os

def load_date(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)  # 返回文件的下一行，在这便是首行，即文件头
        # reader = reader[0:428]
        # 从文件中获取最高温度
        steps, rewards = [], []
        for row in reader:
            # current_date = datetime.strptime(row[1], '%Y-%m-%d')
            step = int(row[1])
            reward = float(row[2])
            steps.append(step)
            rewards.append(reward)

    return [steps, rewards]

def plot(date1, date2, date3, date4):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # for coverage
    plt.xlim(0,3*1e6)
    plt.ylim(40,70)
    plt.xlabel('step')
    plt.ylabel('coverage reward')
    # # for exploration
    # plt.xlim(0,3*1e6)
    # plt.ylim(70,130)
    # plt.xlabel('step')
    # plt.ylabel('exploration reward')

    line1, = ax.plot([], [], color='#D84648', linestyle='-')
    line2, = ax.plot([], [], color='#00C3FE', linestyle='-')
    line3, = ax.plot([], [], color='#FFC43D', linestyle='-')
    line4, = ax.plot([], [], color='#465258', linestyle='-')
    # line5, = ax.plot([], [], color='#700B97', linestyle='-')

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    line1.set_xdata(date1[0])
    line1.set_ydata(date1[1])
    line2.set_xdata(date2[0])
    line2.set_ydata(date2[1])
    line3.set_xdata(date3[0])
    line3.set_ydata(date3[1])
    line4.set_xdata(date4[0])
    line4.set_ydata(date4[1])
    # line5.set_xdata(date5[0])
    # line5.set_ydata(date5[1])
    plt.legend(handles=[line1, line2, line3, line4], labels=['Ours-E_0.1-C_0.9', 'Ours-E_0.2-C_0.8','Ours-E_0.3-C_0.7','Ours-E_0.4-C_0.6'], loc='best')

    plt.grid(axis='y')
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.savefig("/home/zln/repos/adversarial_comms-master/adversarial_comms/test_metric/train_result_coverage.jpg", dpi=800)
    plt.show()

if __name__=='__main__':

    # # # for coverage
    filename1 = '/home/zln/repos/adversarial_comms-master/adversarial_comms/test_metric/train_data/run-0.1e_0.9c_.-tag-ray_tune_evaluation_custom_metrics_reward_coverage_coop_mean.csv'
    filename2 = "/home/zln/repos/adversarial_comms-master/adversarial_comms/test_metric/train_data/run-0.2e_0.8c_.-tag-ray_tune_evaluation_custom_metrics_reward_coverage_coop_mean.csv"
    filename4 = "/home/zln/repos/adversarial_comms-master/adversarial_comms/test_metric/train_data/run-0.3e_0.7c_.-tag-ray_tune_evaluation_custom_metrics_reward_coverage_coop_mean.csv"
    filename3 = "/home/zln/repos/adversarial_comms-master/adversarial_comms/test_metric/train_data/run-0.4e_0.6c_.-tag-ray_tune_evaluation_custom_metrics_reward_coverage_coop_mean.csv"
    
    # # for exploration
    # filename1 = '/home/zln/repos/adversarial_comms-master/adversarial_comms/test_metric/train_data/run-0.1e_0.9c_.-tag-ray_tune_evaluation_custom_metrics_reward_explore_coop_mean.csv'
    # filename2 = "/home/zln/repos/adversarial_comms-master/adversarial_comms/test_metric/train_data/run-0.2e_0.8c_.-tag-ray_tune_evaluation_custom_metrics_reward_explore_coop_mean.csv"
    # filename4 = "/home/zln/repos/adversarial_comms-master/adversarial_comms/test_metric/train_data/run-0.3e_0.7c_.-tag-ray_tune_evaluation_custom_metrics_reward_explore_coop_mean.csv"
    # filename3 = "/home/zln/repos/adversarial_comms-master/adversarial_comms/test_metric/train_data/run-0.4e_0.6c_.-tag-ray_tune_evaluation_custom_metrics_reward_explore_coop_mean.csv"
    date1 = load_date(filename1)
    date2 = load_date(filename2)
    date3 = load_date(filename3)
    date4 = load_date(filename4)
    plot(date1,date2,date4,date3)