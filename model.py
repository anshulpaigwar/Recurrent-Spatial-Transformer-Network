#!/usr/bin/env python
import math

import torch
import torch.nn as nn

from torch.distributions import Normal
from modules import glimpse_network, stn_zoom, context_network
from modules import classification_network, location_network


class RecurrentSpatialTransformer(nn.Module):
    """
    A Recurrent Spatial Transformer Network .

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References
    ----------
    - Minh et. al., https://arxiv.org/abs/1406.6247
    """

    # let us suppose we start with single zoom and learn the zoom instead
    # FIXME: repair input parameters to this function
    def __init__(self):
        """
        Initialize the recurrent attention model and its
        different components.

        Args
        ----
        - g: size of the square patches in the glimpses extracted
          by the retina.
        - k: number of patches to extract per glimpse. (Zoom Values)
        - s: scaling factor that controls the size of successive patches.
        - c: number of channels in each image.
        - h_g: hidden layer size of the fc layer for `phi`.
        - h_l: hidden layer size of the fc layer for `l`.
        - std: standard deviation of the Gaussian policy.
        - hidden_size: hidden size of the rnn.
        - num_classes: number of classes in the dataset.
        - num_glimpses: number of glimpses to take per image,
          i.e. number of BPTT steps.
        """
        super(RecurrentSpatialTransformer, self).__init__()

        # Crop the image using differential STN at location and zoom
        self.stn = stn_zoom(out_height=26, out_width=26)

        # In glimpse Network we convert the cropped image in to feature sapce
        # using CNN and FC Layer then concatenate it with the previous location
        self.sensor = glimpse_network(glimpse_size = 26, h_g = 1024, h_l = 1024, c = 1)
        self.rnn1 = nn.GRUCell(2048, 512)
        self.fc_rnn = nn.Linear(512, 2048)
        self.rnn2 = nn.GRUCell(2048, 512)
        self.classifier = classification_network(input_size = 512, hidden_size = 1024, num_classes = 10)
        self.locator = location_network(input_size = 512, hidden_size = 1024)
        self.context = context_network(100,512)

        # self.baseliner = baseline_network(hidden_size, 1)
        # self.std = std

        # QUESTION: what to do with last flag
    def forward(self, x, loc_prev, ht_prev1, ht_prev2, last=False): # TODO: loop in forward funcion or outside FIXME:parameters
        """
        Run the recurrent attention model for 1 timestep
        on the minibatch of images `x`.

        Args
        ----
        - x: a 4D Tensor of shape (B, H, W, C). The minibatch
          of images.
        - l_t_prev: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the previous
          timestep `t-1`.
        - h_t_prev: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the previous timestep `t-1`.
        - last: a bool indicating whether this is the last timestep.
          If True, the action network returns an output probability
          vector over the classes and the baseline `b_t` for the
          current timestep `t`. Else, the core network returns the
          hidden state vector for the next timestep `t+1` and the
          location vector for the next timestep `t+1`.

        Returns
        -------
        - h_t: a 2D tensor of shape (B, hidden_size). The hidden
          state vector for the current timestep `t`.
        - mu: a 2D tensor of shape (B, 2). The mean that parametrizes
          the Gaussian policy.
        - l_t: a 2D tensor of shape (B, 2). The location vector
          containing the glimpse coordinates [x, y] for the
          current timestep `t`.
        - b_t: a 2D vector of shape (B, 1). The baseline for the
          current time step `t`.
        - log_probas: a 2D tensor of shape (B, num_classes). The
          output log probability vector over the classes.
        """


        
        attention = self.stn(x, loc_prev) # crop the input image according to parameter Theta
        # convolves the croped image convert it to features and fuse it with location information in one vector
        g_t = self.sensor(attention, loc_prev)

        # First Layer of GRU
        ht_1 = self.rnn1(g_t, ht_prev1)

        # Second layer of GRU
        ht_2 = self.fc_rnn(ht_1)
        ht_2 = self.rnn1(ht_2,ht_prev2)

        log_probas = self.classifier(ht_1) # (B,10)
        loc_estimate = self.locator(ht_2) # (B,6)

        return log_probas,loc_estimate,ht_1,ht_2  #FIXME: change the return value








        # ht_prev1 = ht_1
        # ht_prev2 = ht_2
        # theta_prev = theta

        # mu, l_t = self.locator(h_t)
        # b_t = self.baseliner(h_t).squeeze()
        #
        # log_pi = Normal(mu, self.std).log_prob(l_t)
        # log_pi = torch.sum(log_pi, dim=1)
        #
        # if last:
        #     log_probas = self.classifier(h_t)
        #     return h_t, l_t, b_t, log_probas, log_pi
