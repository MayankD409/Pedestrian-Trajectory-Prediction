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




class ConvTemporalGraphical(nn.Module):
    #Source : https://github.com/yysijie/st-gcn/blob/master/net/st_gcn.py
    r"""The basic module for applying a graph convolution.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias)

    def forward(self, x, A):
        assert A.size(0) == self.kernel_size
        x = self.conv(x)
        x = torch.einsum('nctv,tvw->nctw', (x, A))
        return x.contiguous(), A
    

class st_gcn(nn.Module):
    r"""Applies a spatial temporal graph convolution over an input graph sequence.
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``
    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format
        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()
        
#         print("outstg",out_channels)

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,
                                         kernel_size[1])
        

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A):

        res = self.residual(x)
        x, A = self.gcn(x, A)

        x = self.tcn(x) + res
        
        if not self.use_mdn:
            x = self.prelu(x)

        return x, A

class bi_lstm(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(bi_lstm, self).__init__()
        self.hidden_size = hidden_size
        
        # Bi-LSTM layer
        self.bilstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        
        # Output fully connected layer for final concatenation
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional
        
    def forward(self, input_robot):
        # input_robot shape: (batch_size, total_seq_len, input_size)
        input_robot = input_robot.permute(0, 2, 1)
        # Forward pass through Bi-LSTM
        lstm_o, (h_n, _) = self.bilstm(input_robot)

        # Extract final hidden states and concatenate for both directions
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)  # Concatenate hidden states from both directions
        h_n = h_n.unsqueeze(0)
        # Fully connected layer for final output
        output = self.fc(h_n)  # Apply fully connected layer
        
        return output

class biGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(biGRU, self).__init__()
        self.hidden_size = hidden_size
        
        # Bi-GRU layer
        self.bigru = nn.GRU(input_size, hidden_size, bidirectional=True)
        
        # Output fully connected layer for final concatenation
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional
        
    def forward(self, input_robot):
        # input_robot shape: (batch_size, total_seq_len, input_size)
        input_robot = input_robot.permute(0, 2, 1)
        # if input_robot.shape != torch.Size([1, 20, 2]):
        #     print("Input shape is:", input_robot.shape)
        #     print("Input is:", input_robot.shape)
        gru_o, h_n = self.bigru(input_robot)

        # Extract final hidden states and concatenate for both directions
        h_n = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)  # Concatenate hidden states from both directions
        h_n = h_n.unsqueeze(0)

        # Fully connected layer for final output
        output = self.fc(h_n)  # Apply fully connected layer
        
        return output

class biRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(biRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Bi-directional RNN layer
        self.birnn = nn.RNN(input_size, hidden_size, bidirectional=True)
        
        # Output fully connected layer
        self.fc = nn.Linear(hidden_size * 2, output_size)  # *2 for bidirectional
        
    def forward(self, input_robot):
        # input_robot shape: (batch_size, total_seq_len, input_size)
        input_robot = input_robot.permute(0, 2, 1)
        rnn_out, h_n = self.birnn(input_robot)

        # Extract hidden states from both directions
        h_n = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)

        # Fully connected layer
        output = self.fc(h_n)
        return output

    
class social_stgcnn(nn.Module):
    def __init__(self, n_stgcnn, n_txpcnn, input_feat=2, output_feat=5,
                 seq_len=8, pred_seq_len=12, kernel_size=3):
        super(social_stgcnn, self).__init__()
        self.n_stgcnn = n_stgcnn
        self.n_txpcnn = n_txpcnn
        self.seq_len = seq_len
        self.pred_seq_len = pred_seq_len
        self.bilstm = bi_lstm(input_feat, hidden_size=32, output_size=output_feat)
        self.bigru = biGRU(input_feat, hidden_size=16, output_size=output_feat)
        self.birnn = biRNN(input_feat, hidden_size=32, output_size=output_feat)
        # ST-GCNN layers
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat, output_feat, (kernel_size, seq_len)))
        for _ in range(1, self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat, output_feat, (kernel_size, seq_len)))

        # TXP-CNN layers
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(seq_len + seq_len, pred_seq_len, kernel_size=3, padding=1))
        for _ in range(1, self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len, pred_seq_len, kernel_size=3, padding=1))
        self.tpcnn_output = nn.Conv2d(pred_seq_len, pred_seq_len, kernel_size=3, padding=1)

        self.prelus = nn.ModuleList()
        for _ in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())

        self.conv1 = nn.Conv2d(pred_seq_len+pred_seq_len, pred_seq_len, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(pred_seq_len, pred_seq_len, kernel_size=3, padding=1)
        self.prelu = nn.PReLU()

    def forward(self, v, a, robot_trajectory):
        # Process robot trajectory through Bi-LSTM for entire sequence length (seq_len + pred_seq_len)
        robot_encodings = self.bigru(robot_trajectory)
        bi = self.bilstm(robot_trajectory)
        # robot_encodings = self.birnn(robot_trajectory)
        # Split robot encodings into past (seq_len) and future (pred_seq_len) parts
        robot_past_encodings = robot_encodings[:, :self.seq_len, :]
        robot_past_encodings = robot_past_encodings.view(robot_past_encodings.shape[0],robot_past_encodings.shape[2],robot_past_encodings.shape[1])
        robot_future_encodings = robot_encodings[:, self.seq_len:, :]
        robot_future_encodings = robot_future_encodings.view(robot_future_encodings.shape[0],robot_future_encodings.shape[2],robot_future_encodings.shape[1])

        # Process pedestrian trajectory through ST-GCNN
        for k in range(self.n_stgcnn):
            v, a = self.st_gcns[k](v, a)

        # Concatenate ST-GCNN output with past robot encodings
        combined_features = torch.cat((v, robot_past_encodings.unsqueeze(3).repeat(1, 1, 1, v.size(3))), dim=2)

        x = combined_features
        x = x.view(x.shape[0],x.shape[2],x.shape[1],x.shape[3])

        for k in range(self.n_txpcnn):
            x = self.prelus[k](self.tpcnns[k](x))
        
        # Final TXP-CNN output
        final_output = self.tpcnn_output(x)
        final_output = final_output.view(final_output.shape[0],final_output.shape[2],final_output.shape[1],final_output.shape[3])

        # Add future robot encodings to the final output
        # final_output += robot_future_encodings.unsqueeze(3).repeat(1,1,1,final_output.size(3))
        final_output=torch.cat((final_output, robot_future_encodings.unsqueeze(3).repeat(1,1,1,final_output.size(3))), dim=2)
        # print("FInal_output:", final_output.shape)
        final_output = final_output.view(final_output.shape[0],final_output.shape[2],final_output.shape[1],final_output.shape[3])

        # Additional processing layers
        x = self.prelu(self.conv1(final_output))
        x = self.conv2(x)
        x = x.view(x.shape[0],x.shape[2],x.shape[1],x.shape[3])
        return x, a