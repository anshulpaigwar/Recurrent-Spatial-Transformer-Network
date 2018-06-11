#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import numpy as np


def _out_size(self, input_size, kernel_size, stride = 1, padding = 0, pool = False,  pool_kernel_size = 2):
    out_size = (input_size - kernel_size + 2 * padding)/stride + 1
    # flat_features = output_size * output_size * channel
    if pool:
        out_size = out_size/pool_kernel_size
    return int(out_size)


# class EDRAM_Loss(torch.nn.Module):
#
#     def __init__(self):
#         super(EDRAM_Loss,self).__init__()
#
#     def forward(self,log_probas,loc_estimate, target, loc_true):
#         lossEntropy = nn.CrossEntropyLoss()
#         lossMSE = nn.MSELoss()
#         # beta = torch.FloatTensor([[1], [1]])
#         # beta = Variable(beta.cuda())
#         loss_where = lossMSE(loc_estimate , loc_true) # dot product output (B,1)
#         # loss_where = torch.matmul(((loc_estimate - loc_true)**2),beta) # dot product output (B,1)
#         loss_where = torch.mean(loss_where)
#         target = target.view(target.size(0))
#         #print(target.shape)
#         loss_what = lossEntropy(log_probas, target.long()) #(B,1)
#         loss_sum = torch.mean(loss_where + loss_what)
#         # loss_sum = 0.7*loss_where + 0.3*loss_what
#         # return loss_what
#         # return loss_where
#         return loss_sum # TODO: Normalise the loss
#


class EDRAM_Loss(torch.nn.Module):

    def __init__(self):
        super(EDRAM_Loss,self).__init__()

    def forward(self,log_probas,loc_estimate, target, loc_true):
        lossEntropy = nn.CrossEntropyLoss()
        beta = torch.FloatTensor([[1], [0.5], [1], [0.5], [1], [1]])
        beta = Variable(beta.cuda())
        loss_where = torch.matmul(((loc_estimate - loc_true)**2),beta) # dot product output (B,1)
        target = target.view(target.size(0))
        #print(target.shape)
        loss_what = lossEntropy(log_probas, target.long()) #(B,1)
        loss_sum = torch.mean(loss_where + loss_what) # QUESTION: keepdim true??
        return loss_sum
        # return loss_what










class stn_zoom(nn.Module):
    """
    Spatial Transformer network used as differential attention mechanism
    to crop the relevant part of the image.

    Spatial Transformer operation apply on images given the zoom and the location

    Args:
        x (Variable): input images (B x C x H x W)
        l_t_prev: transformation matrix (B x 2 x 3)
        loc (Variable): location of the focus point -- height and width location (B, 2)
        zoom (Variable): zoom for each image -- zoom for height and width (B, 2)
        out_height (int): height output size
        out_width (int): width output size
    Returns:
        grid_sample (Variable): output Tensor of size (B x C x H' x W')
    """

    def __init__(self, out_height, out_width):
        super(stn_zoom, self).__init__()
        self.out_height = out_height
        self.out_width = out_width
        # self.theta = torch.zeros(B, 2, 3)



    def forward(self, x, theta_prev):

        # x.size() = (B, C, W, H)
        # x saves all the images get the batch size by finding size of tensor in 0 direction
        B = x.size(0) # Batch Size
        C = x.size(1) # channels
        output_size = torch.Size((B, C, self.out_height, self.out_width))
        theta_prev = theta_prev.contiguous()
        theta_prev = theta_prev.view(B,2,-1)

        #print ("theta value", theta_prev[25])
        # Get the affine grid (2D flow fields)
        affine_grid = F.affine_grid(theta_prev, output_size) # (B, W, H, 2) and theta.size = (B x 2 x 3)
        # Grid sample
        grid_sample = F.grid_sample(x, affine_grid)

        return grid_sample








class glimpse_network(nn.Module):
    """
    A network that combines the "what" and the "where"
    into a glimpse feature vector `g_t`.

    - "what": glimpse extracted from the retina.
    - "where": location tuple where glimpse was extracted.

    Concretely, feeds the output of the retina `phi` to
    a fc layer and the glimpse location vector `l_t_prev`
    to a fc layer. Finally, these outputs are fed each
    through a fc layer and their sum is rectified.

    In other words:

        `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`

    Args
    ----
    - h_g: hidden layer size of the fc layer for `phi`.
    - h_l: hidden layer size of the fc layer for `l`.
    - glimpse_size: size of the square patches in the glimpses extracted
      by the retina.
    - c: number of channels in each image.
    - phi: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.
    - l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
      coordinates [x, y] for the previous timestep `t-1`.

    Returns
    -------
    - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
      representation returned by the glimpse network for the
      current timestep `t`.
    """

    # TODO:  change all the convolution layer according to the data in paper

    def __init__(self, h_g, h_l, c = 1, glimpse_size = 26): #TODO: repair this
        super(glimpse_network, self).__init__()

        self.conv_drop = nn.Dropout2d()

        self.conv1 = nn.Conv2d(c, 64, kernel_size=3, padding = 1)
        self.conv1_bn = nn.BatchNorm2d(64)

        img_size = _out_size(self, glimpse_size, kernel_size=3, padding = 1) #img_size = 26

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3) #img_size = 24
        self.conv2_bn = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 0)

        img_size = _out_size(self, img_size,kernel_size=3, padding = 0, pool = True)  #img_size = 12

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding = 1) #img_size = 12
        self.conv3_bn = nn.BatchNorm2d(128)

        img_size = _out_size(self, img_size, kernel_size=3, padding = 1)  #img_size = 12


        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding = 1)  #img_size = 12
        self.conv4_bn = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 0)  #img_size = 6

        img_size = _out_size(self, img_size,kernel_size=3, padding = 1, pool = True) #img_size = 6

        self.conv5 = nn.Conv2d(128, 160, kernel_size=3, padding = 1)  #img_size = 6
        self.conv5_bn = nn.BatchNorm2d(160)

        img_size = _out_size(self, img_size, padding = 1, kernel_size=3)   #img_size = 6

        self.conv6 = nn.Conv2d(160, 192, kernel_size=3)   #img_size = 4
        self.conv6_bn = nn.BatchNorm2d(192)

        img_size = _out_size(self, img_size,kernel_size=3, pool = False)

        # glimpse layer
        D_in = img_size * img_size * 192 # 3072
        self.fc1 = nn.Linear(int(D_in), h_g)

        # location layer
        D_in = 6
        self.fc2 = nn.Linear(int(D_in), h_l)

        # make the size of hidden layer of both the FC layer same
        # self.fc3 = nn.Linear(h_g, h_g + h_l) # CHANGED: direct values is subtituted instead of h_l
        # self.fc4 = nn.Linear(h_l, h_g + h_l)
        self.fc3 = nn.Linear(h_g, 512) # CHANGED: direct values is subtituted instead of h_l
        self.fc4 = nn.Linear(h_l, 512)



    def forward(self, phi, l_t_prev):


        # flatten location vector
        l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)

        # Batch Normalise every layer ??
        phi = F.relu(self.conv1_bn(self.conv1(phi)))
        phi = F.relu(self.pool2(self.conv2_bn(self.conv2(phi))))

        phi = F.relu(self.conv3_bn(self.conv3(phi)))
        phi = F.relu(self.pool4(self.conv4_bn(self.conv4(phi))))

        phi = F.relu(self.conv5_bn(self.conv5(phi)))
        phi = F.relu((self.conv6_bn(self.conv6(phi))))


        # Flatten up the image
        phi = phi.view(phi.shape[0], -1)

        # feed phi and l to respective fc layers
        phi_out = F.relu(self.fc1(phi))
        l_out = F.relu(self.fc2(l_t_prev))

        # Batch Normalise FC Layer or Not??
        what = self.fc3(phi_out)
        where = self.fc4(l_out)

        # feed to fc layer
        g_t = F.relu(what) # QUESTION: multiplication or addition

        return g_t


























# class glimpse_network(nn.Module):
#     """
#     A network that combines the "what" and the "where"
#     into a glimpse feature vector `g_t`.
#
#     - "what": glimpse extracted from the retina.
#     - "where": location tuple where glimpse was extracted.
#
#     Concretely, feeds the output of the retina `phi` to
#     a fc layer and the glimpse location vector `l_t_prev`
#     to a fc layer. Finally, these outputs are fed each
#     through a fc layer and their sum is rectified.
#
#     In other words:
#
#         `g_t = relu( fc( fc(l) ) + fc( fc(phi) ) )`
#
#     Args
#     ----
#     - h_g: hidden layer size of the fc layer for `phi`.
#     - h_l: hidden layer size of the fc layer for `l`.
#     - glimpse_size: size of the square patches in the glimpses extracted
#       by the retina.
#     - c: number of channels in each image.
#     - phi: a 4D Tensor of shape (B, H, W, C). The minibatch
#       of images.
#     - l_t_prev: a 2D tensor of shape (B, 2). Contains the glimpse
#       coordinates [x, y] for the previous timestep `t-1`.
#
#     Returns
#     -------
#     - g_t: a 2D tensor of shape (B, hidden_size). The glimpse
#       representation returned by the glimpse network for the
#       current timestep `t`.
#     """
#
#     # TODO:  change all the convolution layer according to the data in paper
#
#     def __init__(self, h_g, c = 1, glimpse_size = 26): #TODO: repair this
#         super(glimpse_network, self).__init__()
#
#         self.conv_drop = nn.Dropout2d()
#
#         self.conv1 = nn.Conv2d(c, 64, kernel_size=3, padding = 1)
#         self.conv1_bn = nn.BatchNorm2d(64)
#
#         img_size = _out_size(self, glimpse_size, kernel_size=3, padding = 1) #img_size = 26
#
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3) #img_size = 24
#         self.conv2_bn = nn.BatchNorm2d(64)
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 0)
#
#         img_size = _out_size(self, img_size,kernel_size=3, padding = 0, pool = True)  #img_size = 12
#
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding = 1) #img_size = 12
#         self.conv3_bn = nn.BatchNorm2d(128)
#
#         img_size = _out_size(self, img_size, kernel_size=3, padding = 1)  #img_size = 12
#
#
#         self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding = 1)  #img_size = 12
#         self.conv4_bn = nn.BatchNorm2d(128)
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 0)  #img_size = 6
#
#         img_size = _out_size(self, img_size,kernel_size=3, padding = 1, pool = True) #img_size = 6
#
#         self.conv5 = nn.Conv2d(128, 160, kernel_size=3, padding = 1)  #img_size = 6
#         self.conv5_bn = nn.BatchNorm2d(160)
#
#         img_size = _out_size(self, img_size, padding = 1, kernel_size=3)   #img_size = 6
#
#         self.conv6 = nn.Conv2d(160, 192, kernel_size=3)   #img_size = 4
#         self.conv6_bn = nn.BatchNorm2d(192)
#
#         img_size = _out_size(self, img_size,kernel_size=3, pool = False)
#
#         # glimpse layer
#         D_in = img_size * img_size * 192 # 3072
#         self.fc1 = nn.Linear(int(D_in), h_g)
#
#
#     def forward(self, phi, l_t_prev):
#
#         # flatten location vector
#         l_t_prev = l_t_prev.view(l_t_prev.size(0), -1)
#
#         # Batch Normalise every layer ??
#         phi = F.relu(self.conv1_bn(self.conv1(phi)))
#         phi = F.relu(self.pool2(self.conv2_bn(self.conv2(phi))))
#
#         phi = F.relu(self.conv3_bn(self.conv3(phi)))
#         phi = F.relu(self.pool4(self.conv4_bn(self.conv4(phi))))
#
#         phi = F.relu(self.conv5_bn(self.conv5(phi)))
#         phi = F.relu((self.conv6_bn(self.conv6(phi))))
#
#         # Flatten up the image
#         phi = phi.view(phi.shape[0], -1)
#
#         # feed phi and l to respective fc layers
#         phi_out = F.relu(self.fc1(phi))
#
#         return phi_out #(B,512)
#
#

class classification_network(nn.Module):

    def __init__(self, input_size = 512, hidden_size = 1024, num_classes = 10):
        super(classification_network, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024,num_classes)

    def forward(self, ht_1):
        a_t = F.relu(self.bn1(self.fc1(ht_1)))
        a_t = F.relu(self.bn2(self.fc2(a_t)))
        a_t = self.fc3(a_t)
        return a_t


class location_network(nn.Module):

    def __init__(self, input_size = 1024, hidden_size = 512, out_size = 6):
        super(location_network, self).__init__()
        # self.std = std
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size,out_size)

    def forward(self, ht_2):
        t = F.relu(self.fc1(ht_2))
        t = F.tanh(self.fc2(t)) # bound between [-1, 1]
        t_p = t.clone()
        t_p[:,0] = torch.clamp(t[:,0],min=0.0, max=1.0) # QUESTION: The clamp operation is not diffrentiable does it backpropogates the error properly??
        t_p[:,4] = torch.clamp(t[:,4],min=0.0, max=1.0)
        # t_p[:,1] = torch.clamp(t[:,0],min=0.0, max=0.0)
        # t_p[:,3] = torch.clamp(t[:,0],min=0.0, max=0.0)
        # torch.clamp(t_p[:,0],min=0.0, max=1.0)
        # torch.clamp(t_p[:,4],min=0.0, max=1.0)
        # prevent gradient flow
        # l_t = l_t.detach()

        return t





# class location_network(nn.Module):
#
#     def __init__(self, input_size = 1024, hidden_size = 1024, out_size = 6):
#         super(location_network, self).__init__()
#         self.std = 0.17
#         # self.fc1 = nn.Linear(input_size, 1024)
#         # self.fc2 = nn.Linear(1024,2)
#         self.fc2 = nn.Linear(input_size,out_size)
#
#
#     def forward(self, ht_2):
#         mu = F.tanh(self.fc2(ht_2)) # bound between [-1, 1]
#
#         # sample from gaussian parametrized by this mean
#         noise = torch.from_numpy(np.random.normal(
#             scale=self.std, size=mu.shape)
#         )
#         noise = Variable(noise.float()).type_as(mu)
#         l_t = mu + noise
#
#         # bound between [-1, 1]
#         l_t = F.tanh(l_t)
#
#         # prevent gradient flow
#         # l_t = l_t.detach()
#
#         return mu
#



























#TODO: resize the input image

class context_network(nn.Module):
    """
    The context network receives a down-sampled lowresolution
    image as input and processes it through a threelayered convolutional
    neural network. It produces a feature vector r(2)
    0 that serves as an initialization of a hidden state
    of the second GRU unit in the recurrent network

    Args
    ----
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.

    Returns
    -------
    - g_t: a 2D tensor of shape (B, hidden_size).
    """

    # TODO:  change all the convolution layer according to the data in paper

    def __init__(self, input_size = 100, hidden_size = 512): #TODO: repair this
        super(context_network, self).__init__()

        self.conv_drop = nn.Dropout2d()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5) #img_size = 96
        self.conv1_bn = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=4, stride=4, padding = 0)  #img_size = 24
        img_size = _out_size(self, input_size, kernel_size = 5, padding = 0, pool = True, pool_kernel_size = 4)

        self.conv2 = nn.Conv2d(16, 16, padding = 1, kernel_size=3) #img_size = 24
        self.conv2_bn = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 0)  #img_size = 12
        img_size = _out_size(self, img_size, padding = 1, kernel_size=3, pool = True, pool_kernel_size = 2 )

        self.conv3 = nn.Conv2d(16, 32, padding = 1, kernel_size=3) #img_size = 12
        self.conv3_bn = nn.BatchNorm2d(32)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding = 0)  #img_size = 6
        img_size = _out_size(self, img_size, kernel_size=3, padding = 1, pool = True, pool_kernel_size = 2)
        D_in = img_size * img_size * 32
        self.fc1 = nn.Linear(int(D_in), hidden_size)




    def forward(self, phi):

        # Batch Normalise every layer ??
        phi = F.relu(self.pool1(self.conv1_bn(self.conv1(phi))))
        phi = F.relu(self.pool2(self.conv2_bn(self.conv2(phi))))
        phi = F.relu(self.pool3(self.conv3_bn(self.conv3(phi))))

        # Flatten up the image
        phi = phi.view(phi.shape[0], -1)

        # feed phi and l to respective fc layers
        phi_out = F.relu(self.fc1(phi))
        # phi_out = F.normalize(phi_out, p=2, dim=1)

        return phi_out





class context_network_2(nn.Module):
    """
    The context network receives a down-sampled lowresolution
    image as input and processes it through a threelayered convolutional
    neural network. It produces a feature vector r(2)
    0 that serves as an initialization of a hidden state
    of the second GRU unit in the recurrent network

    Args
    ----
    - x: a 4D Tensor of shape (B, H, W, C). The minibatch
      of images.

    Returns
    -------
    - g_t: a 2D tensor of shape (B, hidden_size).
    """

    # TODO:  change all the convolution layer according to the data in paper

    def __init__(self, input_size = 12, hidden_size = 1024): #TODO: repair this
        super(context_network_2, self).__init__()

        self.conv_drop = nn.Dropout2d()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=5) #img_size = 8
        self.conv1_bn = nn.BatchNorm2d(16)
        img_size = _out_size(self, input_size, kernel_size = 5, padding = 0, pool = False, pool_kernel_size = 4)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3) #img_size = 6
        self.conv2_bn = nn.BatchNorm2d(16)
        img_size = _out_size(self, img_size, padding = 0, kernel_size=3, pool = False, pool_kernel_size = 2 )

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3) #img_size = 4
        self.conv3_bn = nn.BatchNorm2d(32)
        img_size = _out_size(self, img_size, kernel_size=3, padding = 0, pool = False, pool_kernel_size = 2)
        D_in = img_size * img_size * 32
        self.fc1 = nn.Linear(int(D_in), hidden_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)




    def forward(self, phi):

        # Batch Normalise every layer ??
        # phi = F.relu(self.pool1(self.conv1_bn(self.conv1(phi))))
        # phi = F.relu(self.pool2(self.conv2_bn(self.conv2(phi))))
        # phi = F.relu(self.pool3(self.conv3_bn(self.conv3(phi))))

        phi = self.conv1_bn(self.conv1(phi))
        phi = self.conv2_bn(self.conv2(phi))
        phi = self.conv3_bn(self.conv3(phi))

        # Flatten up the image
        phi_out = phi.view(phi.shape[0], -1)

        # feed phi and l to respective fc layers
        # phi_out = self.fc1(phi_out)
        # phi_out = F.normalize(phi_out, p=2, dim=1)

        # print(phi_out)

        return phi_out
