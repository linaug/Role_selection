import os
import random

command = "evaluate_coop"
model_path = '/home/zln/ray_results/MultiPPO_2023-01-25_00-33-02/MultiPPO_coverage_c578f_00000/checkpoint_000270/'
seed = random.randint(0,50)
out_file = '/home/zln/adv_results/search_rescue/'

train_command = str(command)+' '+str(model_path)+' '+'-s'+' '+str(seed)+' '+str(out_file)+' '+'--trials'+' '+str(500)
os.system(train_command) 
