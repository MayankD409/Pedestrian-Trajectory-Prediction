#!/usr/bin/env python3

import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim
import scipy.sparse as sp
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time


def anorm(p1,p2): 
    NORM = math.sqrt((p1[0]-p2[0])**2+ (p1[1]-p2[1])**2)
    if NORM ==0:
        return 0
    return 1/(NORM)
                
def seq_to_graph(seq_,seq_rel,norm_lap_matr = True):
    seq_ = seq_.squeeze()
    seq_rel = seq_rel.squeeze()
    seq_len = seq_.shape[2]
    max_nodes = seq_.shape[0]

    
    V = np.zeros((seq_len,max_nodes,2))
    A = np.zeros((seq_len,max_nodes,max_nodes))
    for s in range(seq_len):
        step_ = seq_[:,:,s]
        step_rel = seq_rel[:,:,s]
        for h in range(len(step_)): 
            V[s,h,:] = step_rel[h]
            A[s,h,h] = 1
            for k in range(h+1,len(step_)):
                l2_norm = anorm(step_rel[h],step_rel[k])
                A[s,h,k] = l2_norm
                A[s,k,h] = l2_norm
        if norm_lap_matr: 
            G = nx.from_numpy_array(A[s,:,:])
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()
            
    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0
def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=8, skip=1, threshold=0.002,
        min_ped=1, delim='\t',norm_lap_matr = True):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.max_peds_in_frame = 0
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        self.norm_lap_matr = norm_lap_matr

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        non_linear_ped = []
        robot_trajectories = []
        for path in all_files:
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in range(0, num_sequences * self.skip + 1, skip):
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
                '''
                curr_seq_data = np.array([
                                        [1, 101, 10.0, 20.0],
                                        [1, 102, 11.0, 21.0],
                                        [2, 101, 12.0, 22.0],
                                        [2, 103, 13.0, 23.0],
                                        [3, 101, 14.0, 24.0],
                                        [3, 102, 15.0, 25.0],
                                        [4, 102, 16.0, 26.0],
                                        [4, 103, 17.0, 27.0],
                                        [5, 101, 18.0, 28.0],
                                        [5, 102, 19.0, 29.0]
                                    ])
                '''
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                self.max_peds_in_frame = max(self.max_peds_in_frame,len(peds_in_curr_seq))
                for ped_as_robot in peds_in_curr_seq:
                    curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2,
                                            self.seq_len))
                    curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                    '''
                    curr_seq = np.array([
                                        [[1, 2, 3, 4, 5],   # Pedestrian 1 trajectory (x-coordinates)
                                        [2, 3, 4, 5, 6]],  # Pedestrian 1 trajectory (y-coordinates)
                                        
                                        [[11, 12, 13, 14, 15],  # Pedestrian 2 trajectory (x-coordinates)
                                        [12, 13, 14, 15, 16]], # Pedestrian 2 trajectory (y-coordinates)
                                        
                                        [[21, 22, 23, 24, 25],  # Pedestrian 3 trajectory (x-coordinates)
                                        [22, 23, 24, 25, 26]]  # Pedestrian 3 trajectory (y-coordinates)
                                    ])
                    '''
                    curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                            self.seq_len))
                    num_peds_considered = 0
                    _non_linear_ped = []
                    # Count occurrences of each pedestrian
                    ped_counts = np.bincount(curr_seq_data[:, 1].astype(int))
                    most_frequent_ped_id = np.argmax(ped_counts)
                    for _, ped_id in enumerate(peds_in_curr_seq):
                        if ped_id == ped_as_robot:
                            continue
                        curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                    ped_id, :]
                        curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                        pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                        pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1
                        if pad_end - pad_front != self.seq_len:
                            continue
                        curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                        curr_ped_seq = curr_ped_seq
                        # Make coordinates relative
                        rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                        rel_curr_ped_seq[:, 1:] = \
                            curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                        _idx = num_peds_considered
                        curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                        curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                        # Linear vs Non-Linear Trajectory
                        _non_linear_ped.append(
                            poly_fit(curr_ped_seq, pred_len, threshold))
                        curr_loss_mask[_idx, pad_front:pad_end] = 1
                        num_peds_considered += 1

                    robot_seq = curr_seq_data[curr_seq_data[:, 1] == most_frequent_ped_id, :]
                    # print("curr_seq_data:", curr_seq_data[:, 1])
                    '''
                    robot_seq:
                            [[  1. 101.  10.  20.]
                            [  2. 101.  12.  22.]
                            [  3. 101.  14.  24.]
                            [  5. 101.  18.  28.]]
                    '''
                    # print("pedestrian_id:",robot_seq[:,1]) ########################################
                    robot_seq = np.transpose(robot_seq[:, 2:])
                    '''
                    Transposed robot_seq[:, 2:]:
                                            [[10. 12. 14. 18.]
                                            [20. 22. 24. 28.]]
                    '''
                    # print("robot_seq:", robot_seq) ##########################################
                    if num_peds_considered > min_ped:
                        non_linear_ped += _non_linear_ped
                        num_peds_in_seq.append(num_peds_considered)
                        loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                        seq_list.append(curr_seq[:num_peds_considered])
                        seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                        robot_trajectories.append(robot_seq)
                                  
        self.num_seq = len(seq_list)
        print("num:", self.num_seq)
        seq_list = np.concatenate(seq_list, axis=0)
        print("seq_list_shape:", seq_list.shape)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        # print("robot_trajectories before concatenate:", len(robot_trajectories)) ####################################
        # print("robot_trajectories before concatenate:", robot_trajectories[0]) ##########################################
        robot_trajectories = np.array(robot_trajectories)
        # print("robot_trajectories before concatenate:", robot_trajectories) ##################################
        # print("robot_trajectories after concatenate:", robot_trajectories[0]) #######################################################
        # print("robot_trajectories after concatenate:", robot_trajectories[1]) ###############################################
        # print("robot_trajectories after concatenate:", robot_trajectories.shape) #############################
        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len:]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.robot_traj = torch.from_numpy(robot_trajectories).type(torch.float)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        #Convert to Graphs 
        self.v_obs = [] 
        self.A_obs = [] 
        self.v_pred = [] 
        self.A_pred = [] 
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)

            start, end = self.seq_start_end[ss]

            v_,a_ = seq_to_graph(self.obs_traj[start:end,:],self.obs_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            v_,a_=seq_to_graph(self.pred_traj[start:end,:],self.pred_traj_rel[start:end, :],self.norm_lap_matr)
            self.v_pred.append(v_.clone())
            self.A_pred.append(a_.clone())
        pbar.close()

        # Print shapes of final data structures
        print("Shapes of final data structures:")
        print("obs_traj shape:", self.obs_traj.shape) # The shape of obs_traj is assumed to be (num_peds_in_seq, 2, obs_len)
        print("pred_traj shape:", self.pred_traj.shape)
        print("obs_traj_rel shape:", self.obs_traj_rel.shape)
        print("pred_traj_rel shape:", self.pred_traj_rel.shape)
        print("loss_mask shape:", self.loss_mask.shape)
        print("non_linear_ped shape:", self.non_linear_ped.shape)
        print("robot_traj length:", self.robot_traj.shape)
        print("v_obs length:", len(self.v_obs))
        print("A_obs length:", len(self.A_obs))
        print("v_pred length:", len(self.v_pred))
        print("A_pred length:", len(self.A_pred))

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]

        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.pred_traj_rel[start:end, :],
            self.non_linear_ped[start:end], self.loss_mask[start:end, :],
            self.v_obs[index], self.A_obs[index],
            self.v_pred[index], self.A_pred[index]
        ]
        out.extend([self.robot_traj[index]])
        # print("robot_traj shape:", self.robot_traj[index].shape)
        return out

# The shape of the robot_traj should be [num_sequences, 2 (for x,y), 20(no. of time pedestrian accours basically)]