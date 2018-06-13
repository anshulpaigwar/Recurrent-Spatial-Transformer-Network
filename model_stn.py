#!/usr/bin/env python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from modules import glimpse_network, stn_zoom, context_network, context_network_2
from modules import classification_network, location_network
import ipdb as pdb

class RecurrentSpatialTransformer(nn.Module):

    def __init__(self):

        super(RecurrentSpatialTransformer, self).__init__()

        self.context = context_network(100,1024)
        self.rnn = nn.GRU(1024,512,1)
        self.locator = location_network(input_size = 512, hidden_size = 1024)

        # Crop the image using differential STN at location and zoom
        self.stn = stn_zoom(out_height=26, out_width=26)
        self.sensor = glimpse_network(glimpse_size = 26, h_g = 512,h_l = 1024, c = 1)
        self.classifier = classification_network(input_size = 512, hidden_size = 1024, num_classes = 10)
        self.context_2 = context_network_2(12,1024)



    def forward(self, x, x_resized, h0,seq_len):

        # in_seq = self.context(x) # (B,512)
        # # print(in_seq[1])
        #
        # in_seq = in_seq.unsqueeze(0) #(1,B,512)
        # in_seq = in_seq.expand(seq_len,-1,-1) #(6,B,512)
        # # print(in_seq.shape)
        # # print(in_seq[2,1])
        # # pdb.set_trace()
        # out_seq, hn = self.rnn(in_seq,h0) # out_seq.shape = (6,B,1024)
        #
        # glimpse_list = []
        # loc_estimate_list = []
        # output = []
        # for i in range(seq_len):
        #     # print(out_seq[i,1])
        #
        #     loc_estimate = self.locator(out_seq[i]) # (B,6)
        #     loc_estimate_list.append(loc_estimate)
        #     attention = self.stn(x, loc_estimate) # (B,26,26) crop the input image according to parameter Theta
        #     glimpse_list.append(attention[1])
        #     g_t = self.sensor(attention, loc_estimate)
        #     log_probas = self.classifier(g_t) # (B,10)
        #     output.append(log_probas)
        # output = torch.stack(output)
        # glimpse_list = torch.stack(glimpse_list)
        # loc_estimate_list = torch.stack(loc_estimate_list)
        # return output, glimpse_list,loc_estimate_list



        in_seq = self.context_2(x_resized) # (B,512)
        # in_seq = self.context(x) # (B,512)
        # print(in_seq[1])

        in_seq = in_seq.unsqueeze(0) #(1,B,512)
        in_seq = in_seq.expand(seq_len,-1,-1) #(6,B,512)
        # print(in_seq.shape)
        # print(in_seq[2,1])
        # pdb.set_trace()
        out_seq, hn = self.rnn(in_seq,h0) # out_seq.shape = (6,B,1024)

        glimpse_list = []
        loc_estimate_list = []
        output = []
        out_seq = F.normalize(out_seq, p=2, dim=2)
        for i in range(seq_len):
            # print(out_seq[i,1])

            loc_estimate = self.locator(out_seq[i]) # (B,6)
            loc_estimate_list.append(loc_estimate)
            attention = self.stn(x, loc_estimate) # (B,26,26) crop the input image according to parameter Theta
            glimpse_list.append(attention[1])
            g_t = self.sensor(attention, loc_estimate)
            log_probas = self.classifier(g_t) # (B,10)
            output.append(log_probas)
        output = torch.stack(output)
        glimpse_list = torch.stack(glimpse_list)
        loc_estimate_list = torch.stack(loc_estimate_list)
        return output, glimpse_list,loc_estimate_list
