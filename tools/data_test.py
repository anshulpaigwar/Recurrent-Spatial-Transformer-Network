#!/usr/bin/env python
import numpy as np
import h5py
import torch
import torchvision
import torch.utils.data as data
import torch.multiprocessing
# torch.multiprocessing.set_start_method('spawn')
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib
import matplotlib.pyplot as plt
from skimage.transform import resize

# #data_dir = "/home/anshul/inria_thesis/datasets/mnist_sequence3_sample_8distortions_9x9.npz"
# data_dir = "/home/anshul/inria_thesis/datasets/mnist_sequence1_sample_5distortions5x5.npz"
# data = np.load(data_dir)
# print(data['y_train'])

# def keys(f):
#     return [key for key in f.keys()]
#
# data_dir = "/home/anshul/inria_thesis/datasets/mnist-cluttered/mnist_clusttered.hdf5"
# f = h5py.File(data_dir, 'r')
# print(keys(f))
# data = f['features']
# target = f['labels']
# loc = f['locations']
#
# print(data.shape)
# print(target.shape)
# print(loc.shape)





def imshow(img, labels):
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    print(npimg.shape)
    #plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.imshow(npimg[1], cmap='gray', interpolation='none')
    plt.title(' '.join('%f' % labels[1,j] for j in range(len(labels[1]))), fontsize=20)
    plt.show()









class H5Dataset(data.Dataset):

    def __init__(self, file_path, transform = None):
        super(H5Dataset, self).__init__()
        f = h5py.File(file_path)
        # self.split = split
        self.data = f['features']
        self.target = f['labels']
        self.loc = f['locations']
        self.transform = transform
        # self.data_resized = []

    def __getitem__(self, index):
        # print(self.data.shape)
        # print(self.target.shape)
        # img = self.data[index,:,:,:]
        label = [float(self.target[index])] # since the target values are 1d array with values of type numpy.uint8 we have to convert them

        if self.transform:
            data_resized = resize(self.data[index,0,:,:], (12, 12),mode =  "symmetric")
            data_resized = torch.from_numpy(data_resized).float()
            data_resized = data_resized.unsqueeze(0)

            # print(data_resized.shape)


        return (#torch.from_numpy(img[0]).float(),
                torch.from_numpy(self.data[index,:,:,:]).float(),
                data_resized,
                torch.FloatTensor(label).float(),
                torch.from_numpy(self.loc[index,:]).float())

    def __len__(self):
        return self.data.shape[0] # FIXME: return according to split value





def get_data_loaders(data_dir,
                    show_sample=True):

    datasets = H5Dataset(data_dir,transform = True)
    data_len = len(datasets)
    print("Total Data size ", data_len)
    indices = list(range(data_len))

    # TODO: Learn to make a config file and input parameters from that
    # TODO: Learn to use kwargs

    shuffle = False
    random_seed = 20
    # batch_size = 64

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("using cuda")
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 4
        pin_memory = True


    # data_loader = torch.utils.data.DataLoader(datasets,batch_size=batch_size,num_workers=num_workers, pin_memory=pin_memory)





    # split the main dataset into three parts test(20%), valid(20%) and train (70%)
    split_train = int(np.floor(0.64 * data_len))
    split_valid_test = int(np.floor(0.82 * data_len))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx = indices[:split_train]
    valid_idx = indices[split_train:split_valid_test]
    test_idx = indices[split_valid_test:]
    print("Train Data size ",len(train_idx))
    print("Valid Data size ",len(valid_idx))
    print("Test Data size ",len(test_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    test_sampler = SubsetRandomSampler(test_idx)


    train_loader = torch.utils.data.DataLoader(
        datasets, batch_size= 128, sampler=train_sampler, # BS = 105
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        datasets, batch_size= 32, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets, batch_size= 32, sampler=test_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # show_sample = True

    if(show_sample):

        # get some random training images
        dataiter = iter(train_loader)
        images, labels, loc = dataiter.next()
        # print(len(images))
        # show images
        imshow(torchvision.utils.make_grid(images), labels)

    return train_loader, valid_loader, test_loader
















if __name__ == '__main__':

    data_dir = "/home/anshul/inria_thesis/datasets/mnist-cluttered/mnist_clusttered.hdf5"
    train_loader, valid_loader, test_loader = get_data_loaders(data_dir, show_sample=True)
    print(len(train_loader))
    for batch_idx, (data, target, loc_true) in enumerate(train_loader):
        print(batch_idx)
