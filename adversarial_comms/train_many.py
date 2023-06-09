import os
import random

command = "train_policy"
seed = random.randint(0,9)
timestep_totle = 20

train_command = str(command)+' '+'coverage'+' '+'-s'+' '+str(seed)+' '+'-t'+' '+str(timestep_totle)
os.system(train_command) 
