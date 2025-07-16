#!/usr/bin/env python3

import os
import math
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pickle
import argparse
import glob
import torch.distributions.multivariate_normal as torchdist
from utils import * 
from metrics import * 
from model import social_stgcnn
import copy
import time
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt

import numpy as np
import random

def generate_random_trajectory(start_point, end_point, length=12):
    """
    Generates a random trajectory from start_point to end_point of specified length.
    
    Parameters:
    - start_point: Starting point (x, y) as a torch tensor.
    - end_point: Ending point (x, y) as a torch tensor.
    - length: Number of time steps for the trajectory
    
    Returns:
    - random_trajectory: Generated random trajectory as a torch tensor of shape (2, length)
    """
    random_trajectory = torch.zeros((2, length))
    random_trajectory[:, 0] = start_point
    random_trajectory[:, -1] = end_point

    for t in range(1, length - 1):
        alpha = t / (length - 1)
        random_trajectory[:, t] = (1 - alpha) * start_point + alpha * end_point + torch.randn(2) * 0.1  # Add noise
    
    return random_trajectory

def plot_pedestrian_trajectories(V_y_rel_to_abs, V_pred_rel_to_abs, robot_traj, type, test_no):
    plt.figure(figsize=(10, 8))
    print("robot_traj shape:", robot_traj.shape)
    # Plot robot trajectory
    robot_traj = np.array(robot_traj)  # Convert to numpy array if not already
    plt.plot(robot_traj[0][0][8:20], robot_traj[0][1][8:20], color='blue', linestyle='-', label='Robot Trajectory')
    
    # Mark start and end points for robot trajectory with smaller markers
    plt.scatter(robot_traj[0][0][8], robot_traj[0][1][8], color='green', marker='s', label='Robot Start', s=50)
    plt.scatter(robot_traj[0][0][19], robot_traj[0][1][19], color='red', marker='s', label='Robot End', s=50)

    # Define a list of colors for different pedestrians
    colors = ['black', 'red', 'green', 'magenta', 'cyan', 'yellow', 'purple', 'orange', 'brown', 'pink']
    
    # Plot pedestrian trajectories
    for ped in range(V_pred_rel_to_abs.shape[1]):
        x_traj = V_pred_rel_to_abs[:, ped, 0]  # X locations of pedestrian 'ped'
        y_traj = V_pred_rel_to_abs[:, ped, 1]  # Y locations of pedestrian 'ped'
        plt.plot(x_traj, y_traj, color=colors[ped % len(colors)], linestyle='--', label=f'Pedestrian {ped + 1} Trajectory')
        
        # Mark start and end points for pedestrian trajectories with different colors
        plt.scatter(x_traj[0], y_traj[0], color='yellow', marker='^', label=f'Pedestrian {ped + 1} Start', s=50)
        plt.scatter(x_traj[-1], y_traj[-1], color='grey', marker='^', label=f'Pedestrian {ped + 1} End', s=50)

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Pedestrian and Robot Trajectories')
    plt.legend()
    plt.grid(True)
    
    # Set the same scale for all graphs starting from (0,0)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    
    plt.savefig(f'1lyr_imgs/{type}_{test_no}.png', format='png')
    plt.close()

    return robot_traj[:][:][:9]

def check_collision(robot_traj, V_pred_rel_to_abs):
    # Only check the last 12 timesteps of the robot trajectory and pedestrian predictions
    for t in range(8, 20):
        robot_position = robot_traj[:, :, t]
        ped_t = t - 8  # Adjust the timestep for the pedestrian predictions
        for ped in range(V_pred_rel_to_abs.shape[1]):
            ped_position = V_pred_rel_to_abs[ped_t, ped, :]
            if np.array_equal(robot_position, ped_position):
                print(f'Collision detected at timestep {t} between robot and pedestrian {ped + 1}')
                return True
    return False


def test(KSTEPS=20, constant_robot_traj = None, type = None, test_no = None):
    global loader_test,model
    model.eval()
    ade_bigls = []
    fde_bigls = []
    raw_data_dict = {}
    step =0
    total_time = 0
    
    for batch in loader_test: 
        step+=1
        start_time = time.time()
        #Get data
        # batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
         loss_mask,V_obs,A_obs,V_tr,A_tr, robot_traj = batch


        num_of_objs = obs_traj_rel.shape[1]

        if constant_robot_traj is not None:
            robot_traj = constant_robot_traj
                
        # if step == 9:
        #     print("Robot_traj:", robot_traj)
        
        V_obs_tmp =V_obs.permute(0,3,1,2)

        V_pred,_ = model(V_obs_tmp,A_obs.squeeze(), robot_traj)
        end_time = time.time()
        V_pred = V_pred.permute(0,2,3,1)

        

        V_tr = V_tr.squeeze()
        A_tr = A_tr.squeeze()
        V_pred = V_pred.squeeze()
        num_of_objs = obs_traj_rel.shape[1]
        V_pred,V_tr =  V_pred[:,:num_of_objs,:],V_tr[:,:num_of_objs,:]

        sx = torch.exp(V_pred[:,:,2]) #sx, V_pred[seq,node,feat]
        sy = torch.exp(V_pred[:,:,3]) #sy
        corr = torch.tanh(V_pred[:,:,4]) #corr
        
        cov = torch.zeros(V_pred.shape[0],V_pred.shape[1],2,2)#.cuda()
        cov[:,:,0,0]= sx*sx
        cov[:,:,0,1]= corr*sx*sy
        cov[:,:,1,0]= corr*sx*sy
        cov[:,:,1,1]= sy*sy
        mean = V_pred[:,:,0:2]
        
        mvnormal = torchdist.MultivariateNormal(mean,cov)


        ### Rel to abs 
        ##obs_traj.shape = torch.Size([1, 6, 2, 8]) Batch, Ped ID, x|y, Seq Len 
        
        #Now sample 20 samples
        ade_ls = {}
        fde_ls = {}
        V_x = seq_to_nodes(obs_traj.data.cpu().numpy().copy())
        V_x_rel_to_abs = nodes_rel_to_nodes_abs(V_obs.data.cpu().numpy().squeeze().copy(),
                                                 V_x[0,:,:].copy())

        V_y = seq_to_nodes(pred_traj_gt.data.cpu().numpy().copy())
        V_y_rel_to_abs = nodes_rel_to_nodes_abs(V_tr.data.cpu().numpy().squeeze().copy(),
                                                 V_x[-1,:,:].copy())
        
        raw_data_dict[step] = {}
        raw_data_dict[step]['obs'] = copy.deepcopy(V_x_rel_to_abs)
        raw_data_dict[step]['trgt'] = copy.deepcopy(V_y_rel_to_abs)
        raw_data_dict[step]['pred'] = []

        for n in range(num_of_objs):
            ade_ls[n]=[]
            fde_ls[n]=[]

        for k in range(KSTEPS):

            V_pred = mvnormal.sample()



            #V_pred = seq_to_nodes(pred_traj_gt.data.numpy().copy())
            V_pred_rel_to_abs = nodes_rel_to_nodes_abs(V_pred.data.cpu().numpy().squeeze().copy(),
                                                     V_x[-1,:,:].copy())
            raw_data_dict[step]['pred'].append(copy.deepcopy(V_pred_rel_to_abs))
           # print(V_pred_rel_to_abs.shape) #(12, 3, 2) = seq, ped, location
            for n in range(num_of_objs):
                pred = [] 
                target = []
                obsrvs = [] 
                number_of = []
                pred.append(V_pred_rel_to_abs[:,n:n+1,:])
                target.append(V_y_rel_to_abs[:,n:n+1,:])
                obsrvs.append(V_x_rel_to_abs[:,n:n+1,:])
                number_of.append(1)

                ade_ls[n].append(ade(pred,target,number_of))
                fde_ls[n].append(fde(pred,target,number_of))

            # Check for collisions
            if check_collision(robot_traj, V_pred_rel_to_abs):
                print(f"Collision detected in batch {step} at sample {k}")
            
        # Concatenate the observed and predicted trajectory of pedestrian 4 at step 9
        if step == 12:
            observed_traj = torch.tensor(V_x_rel_to_abs[:, 1, :]).T
            predicted_traj = torch.tensor(V_pred_rel_to_abs[:, 1, :]).T
            pedestrian_4_traj = torch.cat((observed_traj, predicted_traj), dim=1).unsqueeze(0)
            print("pedestrian_4_traj:", pedestrian_4_traj)
            rt = plot_pedestrian_trajectories(V_y_rel_to_abs, V_pred_rel_to_abs, robot_traj, type, test_no)

        
        for n in range(num_of_objs):
            ade_bigls.append(min(ade_ls[n]))
            fde_bigls.append(min(fde_ls[n]))

          # Record end time for batch processing
        batch_time = end_time - start_time
        total_time += batch_time

    ade_ = sum(ade_bigls)/len(ade_bigls)
    fde_ = sum(fde_bigls)/len(fde_bigls)
    avg_inference_time = total_time / len(loader_test)

    return ade_,fde_,raw_data_dict, avg_inference_time, pedestrian_4_traj, rt


paths = ['./checkpoint/1lyr*']
KSTEPS=20

print("*"*50)
print('Number of samples:',KSTEPS)
print("*"*50)




for feta in range(len(paths)):
    ade_ls = [] 
    fde_ls = [] 
    inference_times = []
    path = paths[feta]
    exps = glob.glob(path)
    print('Model being tested are:',exps)

    for exp_path in exps:
        print("*"*50)
        print("Evaluating model:",exp_path)

        model_path = exp_path+'/val_best.pth'
        args_path = exp_path+'/args.pkl'
        with open(args_path,'rb') as f: 
            args = pickle.load(f)

        stats= exp_path+'/constant_metrics.pkl'
        with open(stats,'rb') as f: 
            cm = pickle.load(f)
        print("Stats:",cm)



        #Data prep     
        obs_seq_len = args.obs_seq_len
        pred_seq_len = args.pred_seq_len
        data_set = './datasets/'+args.dataset+'/'

        dset_test = TrajectoryDataset(
                data_set+'test/',
                obs_len=obs_seq_len,
                pred_len=pred_seq_len,
                skip=1,norm_lap_matr=True)

        loader_test = DataLoader(
                dset_test,
                batch_size=1,#This is irrelative to the args batch size parameter
                shuffle =False,
                num_workers=1)



        #Defining the model 
        model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
        output_feat=args.output_size,seq_len=args.obs_seq_len,
        kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len)#.cuda()
        model.load_state_dict(torch.load(model_path))


        ade_ =999999
        fde_ =999999
        print("Testing ....")
        ad,fd,raw_data_dic_, avg_inf_time, pedestrian_4_traj, rt= test(constant_robot_traj = None, type = args.tag, test_no = "a")
        ade_= min(ade_,ad)
        fde_ =min(fde_,fd)
        ade_ls.append(ade_)
        fde_ls.append(fde_)
        inference_times.append(avg_inf_time)
        print("ADE:",ade_," FDE:",fde_, "Average Inference Time per Batch:", avg_inf_time)

        # Extract the first 8 timesteps of the robot trajectory and the end point of the second pedestrian
        robot_traj_start = torch.from_numpy(rt[:, :, :8])  # First 8 timesteps of the robot trajectory
        robot_start_point = robot_traj_start[:, :, -1]  # 8th timestamp as the start point for prediction horizon
        pedestrian_end_point = pedestrian_4_traj[:, :, -1]  # End point of the second pedestrian
        print("rsp:", robot_start_point.shape)
        print("pep:", pedestrian_end_point.shape)

        # Generate a random trajectory for the robot
        random_traj = generate_random_trajectory(start_point=robot_start_point, end_point=pedestrian_end_point)
        random_robot_traj = torch.cat((robot_traj_start.squeeze(), random_traj), dim=1).unsqueeze(0)
        print("rr:", random_robot_traj.shape)

        print("Testing with Pedestrian 2's Observed and Predicted Trajectory as Robot Trajectory...")
        ad,fd,raw_data_dic_, avg_inf_time, pedestrian_4, rp= test(constant_robot_traj = random_robot_traj, type = args.tag, test_no = "b")
        ade_= min(ade_,ad)
        fde_ =min(fde_,fd)
        ade_ls.append(ade_)
        fde_ls.append(fde_)
        inference_times.append(avg_inf_time)
        print("ADE:",ade_," FDE:",fde_, "Average Inference Time per Batch:", avg_inf_time)





    print("*"*50)

    print("Avg ADE:",sum(ade_ls)/5)
    print("Avg FDE:",sum(fde_ls)/5)
    print("Avg Inference Time across Models:", sum(inference_times) / len(exps))