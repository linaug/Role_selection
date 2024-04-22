import os
import random

command = "train_policy"
timestep_totle = 5
# alpha = [0.1,0.2,0.3,0.4]
# alpha = [0.3,0.2]
alpha = [1]

for i in range(len(alpha)):
    train_command = str(command)+' '+'coverage'+' '+'-t'+' '+str(timestep_totle)+' '+'-alpha'+' '+str(alpha[i])
    os.system(train_command) 
