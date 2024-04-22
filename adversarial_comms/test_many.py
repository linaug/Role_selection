import os
import random

command = "evaluate_coop"
# model_path = ['/home/zln/ray_results/Train_explore_loudian_0.1v0.9/MultiPPO_coverage_a418b_00000/checkpoint_000750/',
#               '/home/zln/ray_results/Train_explore_loudian_0.2v0.8/MultiPPO_coverage_3a940_00000/checkpoint_000750/',
#               '/home/zln/ray_results/Train_explore_loudian_0.3v0.7/MultiPPO_coverage_65aad_00000/checkpoint_000750/',
#               '/home/zln/ray_results/Train_explore_loudian_0.4v0.6/MultiPPO_coverage_bf87a_00000/checkpoint_000750/']
# model_path = ['/home/zln/ray_results/Train_explore_cover_re_2/Train_explore_cover_re_2_0.1v0.9/MultiPPO_coverage_91bd9_00000/checkpoint_000620/']
model_path = ['/home/zln/ray_results/Train_explore_cover_re_2/Train_explore_cover_re_2_0.2v0.8/MultiPPO_coverage_49930_00000/checkpoint_000620/',
              '/home/zln/ray_results/Train_explore_cover_re_2/Train_explore_cover_re_2_0.3v0.7/MultiPPO_coverage_a571b_00000/checkpoint_000620/']
# model_path = ['/home/zln/ray_results/Train_explore_cover_re_2/Train_explore_cover_re_2_0.4v0.6/MultiPPO_coverage_e234e_00000/checkpoint_000620/']

out_file = '/home/zln/adv_results/search_rescue/0725_interest_100/train_both_explore_cover_re_2_0925/all_role_pkl/'

# interest_fraction = [0.08,0.16,0.24,0.56]
# coverable_fraction = [0.84,0.6,0.49]
# interest_fraction = [0.16]
# coverable_fraction = [0.84,0.49]
interest_fraction = [0.24]
coverable_fraction = [0.6]

# test for output the graph

for i in range(len(model_path)):
    for j in range(len(interest_fraction)):
        for k in range(len(coverable_fraction)):
            train_command = str(command)+' '+str(model_path[i])+' '+'--out_file'+' '+str(out_file)+' '+"--interest"+' '+str(interest_fraction[j])+' '+"--coverable"+' '+str(coverable_fraction[k])+' '+'--trials'+' '+str(500)
            os.system(train_command) 

