#!/usr/bin/env python
import numpy as np
import torch
import torchvision
import torch.utils.data as data
import torch.multiprocessing
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from skimage.transform import resize
import ipdb as pdb




def visualize(data, fig_num, title):
    # input_tensor = data.cpu()
    # input_tensor = torch.squeeze(input_tensor)
    # in_grid = input_tensor.detach().numpy()

    fig=plt.figure(num = fig_num)
    plt.imshow(data, cmap='gray', interpolation='none')
    plt.title(title)
    figManager = plt.get_current_fig_manager()
    figManager.resize(*figManager.window.maxsize())
    plt.show(block=False)
    # time.sleep(4)
    # plt.close()






class MnistSeq(Dataset):

    def __init__(self, data_dir, train = True, transform = None):
        super(MnistSeq, self).__init__()
        self.train = train
        self.transform = transform

        if self.train:
            print('loading training data ')
            self.train_data = np.load(data_dir + "X_train.npy")
            self.train_data = self.train_data.reshape(-1,100,100)
            self.train_labels = np.load(data_dir + "y_train.npy")
            print("train dataset", len(self.train_data))
            # print(self.train_data.shape)
            # print(len(self.train_labels))
            # print(self.train_labels[2])
            # visualize(self.train_data[2], 1, "train image")

        else:
            print('loading validation data ')
            self.valid_data = np.load(data_dir + "X_valid.npy")
            self.valid_data = self.valid_data.reshape(-1,100,100)
            self.valid_labels = np.load(data_dir + "y_valid.npy")
            print("train dataset", len(self.valid_data))

    def __getitem__(self, index):
    	if self.train:
    	    data, label = self.train_data[index], self.train_labels[index]
    	else:
    	    data, label = self.valid_data[index], self.valid_labels[index]

        if self.transform:
            data_resized = resize(data, (12, 12),mode =  "symmetric")
            # data_resized = torch.from_numpy(data_resized).float()
            # data_resized = data_resized.unsqueeze(0)
            return data, data_resized, label

        return data, label

    def __len__(self):
        if self.train:
            return self.train_data.shape[0]
        else:
            return self.valid_data.shape[0]







def get_data_loaders(data_dir, batch = 32,  transform = True):

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("using cuda")
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 4
        pin_memory = True


    train_loader = DataLoader(MnistSeq(data_dir,train = True, transform = transform),
                    batch_size= batch, num_workers=num_workers, pin_memory=pin_memory,shuffle=True)

    valid_loader = DataLoader(MnistSeq(data_dir,train = False, transform = transform),
                    batch_size= batch, num_workers=num_workers, pin_memory=pin_memory,shuffle=True)

    return train_loader, valid_loader






if __name__ == '__main__':

    data_dir = "/home/anshul/inria_thesis/datasets/mnist_sequence3_sample_8distortions_9x9/"

    # d = MnistSeq(data_dir,train = True, transform = True)
    # pdb.set_trace()

    train_loader, valid_loader = get_data_loaders(data_dir, batch = 32, transform = True)
    print(len(train_loader))
    for batch_idx, (data, data_resized, labels) in enumerate(valid_loader):
        print(data[0])
        print(data_resized[0])
        print(labels[0])
        pdb.set_trace()





'''


        label = [float(self.target[index])] # since the target values are 1d array with values of type numpy.uint8 we have to convert them



            # print(data_resized.shape)


        return (#torch.from_numpy(img[0]).float(),
                torch.from_numpy(self.data[index,:,:,:]).float(),
                data_resized,
                torch.FloatTensor(label).float(),
                torch.from_numpy(self.loc[index,:]).float())

    def __len__(self):
        return self.data.shape[0] # FIXME: return according to split value


'''
