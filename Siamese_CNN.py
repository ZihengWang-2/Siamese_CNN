import codecs
import errno
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.stats
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch import optim
from torchvision import transforms

'''
Select a dataset.
'''
# dataset = 'mnist'
# ataset = 'fashion-mnist'
dataset = 'cifar10'
# dataset = 'SVHN'
# dataset = 'STL10'
# dataset = 'omniglot'


'''
Training Configurations
'''
random_seed = 20
do_learn = False
save_frequency = 2
batch_size = 16
lr = 0.001
num_epochs = 50
weight_decay = 0.0001

'''
Test Configurations
'''
threshold = 0.5
trial = 200


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
    assert get_int(data[:4]) == 2049
    length = get_int(data[4:8])
    parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
    return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
    assert get_int(data[:4]) == 2051
    length = get_int(data[4:8])
    num_rows = get_int(data[8:12])
    num_cols = get_int(data[12:16])
    images = []
    parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
    return torch.from_numpy(parsed).view(length, num_rows, num_cols)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on:
    """

    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, x0, x1, y):
        # euclidian distance
        y = y.float()
        diff = x0 - x1
        dist_sq = torch.sum(torch.pow(diff, 2), 1)
        dist = torch.sqrt(dist_sq + 1e-20)

        mdist = self.margin - dist
        # mdist = self.margin - dist_sq
        dist = torch.clamp(mdist, min=0.0)
        loss = y * dist_sq + (1 - y) * torch.pow(dist, 2)
        # loss = y * dist_sq + (1 - y) * dist
        loss = torch.sum(loss) / 2.0 / x0.size()[0]
        return loss


class BalancedMNISTPair(torch.utils.data.Dataset):
    """Dataset that on each iteration provides two random pairs of
     images. One pair is of the same number (positive sample), one
    is of two different numbers (negative sample).
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]

    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = '-training.pt'
    test_file = '-test.pt'

    def __init__(self, root, dataset, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.dataset = dataset

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.dataset + self.training_file))

            self.class_n = int(torch.max(self.train_labels).item() + 1)
            aaa = torch.max(self.train_data).item()
            train_labels_class = []
            train_data_class = []
            for i in range(self.class_n):
                indices = torch.squeeze(torch.nonzero(self.train_labels == i, as_tuple=False))
                train_labels_class.append(torch.index_select(self.train_labels, 0, indices))
                train_data_class.append(torch.index_select(self.train_data, 0, indices))

            # generate balanced pairs
            self.train_data = []
            self.train_labels = []
            lengths = [x.shape[0] for x in train_labels_class]
            for i in range(self.class_n):
                for j in range(lengths[i] // 2 - 1):  # create 500 pairs
                    rnd_cls = random.randint(0, self.class_n - 2)  # choose random class that is not the same class
                    if rnd_cls >= i:
                        rnd_cls = rnd_cls + 1

                    rnd_dist = random.randint(0, lengths[i] // 2)
                    if j >= lengths[rnd_cls]:
                        self.train_data.append(torch.stack(
                            [train_data_class[i][j], train_data_class[i][lengths[i] // 2 + rnd_dist - 1],
                             train_data_class[rnd_cls][j // 2]]))
                    else:
                        self.train_data.append(torch.stack(
                            [train_data_class[i][j], train_data_class[i][lengths[i] // 2 + rnd_dist - 1],
                             train_data_class[rnd_cls][j]]))
                    self.train_labels.append([1, 0])

            self.train_data = torch.stack(self.train_data)
            self.train_labels = torch.tensor(self.train_labels)

        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.dataset + self.test_file))
            self.class_n = int(torch.max(self.test_labels).item() + 1)

            test_labels_class = []
            test_data_class = []
            for i in range(self.class_n):
                indices = torch.squeeze(torch.nonzero(self.test_labels == i, as_tuple=False))
                test_labels_class.append(torch.index_select(self.test_labels, 0, indices))
                test_data_class.append(torch.index_select(self.test_data, 0, indices))

            # generate balanced pairs
            self.test_data = []
            self.test_labels = []
            lengths = [x.shape[0] for x in test_labels_class]
            for i in range(self.class_n):
                for j in range(lengths[i] // 2 - 1):  # create 500 pairs
                    rnd_cls = random.randint(0, self.class_n - 2)  # choose random class that is not the same class
                    if rnd_cls >= i:
                        rnd_cls = rnd_cls + 1

                    rnd_dist = random.randint(0, lengths[i] // 2)
                    if j >= lengths[rnd_cls]:
                        self.test_data.append(torch.stack(
                            [test_data_class[i][j], test_data_class[i][lengths[i] // 2 + rnd_dist - 1],
                             test_data_class[rnd_cls][j // 2]]))
                    else:
                        self.test_data.append(torch.stack(
                            [test_data_class[i][j], test_data_class[i][lengths[i] // 2 + rnd_dist - 1],
                             test_data_class[rnd_cls][j]]))
                    self.test_labels.append([1, 0])

            self.test_data = torch.stack(self.test_data)
            self.test_labels = torch.tensor(self.test_labels)

    def __getitem__(self, index):
        if self.train:
            imgs, target = self.train_data[index], self.train_labels[index]
        else:
            imgs, target = self.test_data[index], self.test_labels[index]

        img_ar = []
        for i in range(len(imgs)):
            img = Image.fromarray(imgs[i].numpy(), mode='L')
            if self.transform is not None:
                img = self.transform(img)
            img_ar.append(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_ar, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.dataset + self.training_file)) and \
               os.path.exists(os.path.join(self.root, self.processed_folder, self.dataset + self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise
        '''
        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)
        '''

        # process and save as torch files
        print('Processing...')

        if self.dataset == 'mnist' or self.dataset == 'fashion-mnist':
            training_set = (
                read_image_file(
                    os.path.join(self.root, self.raw_folder, self.dataset + '-' + 'train-images-idx3-ubyte')),
                read_label_file(
                    os.path.join(self.root, self.raw_folder, self.dataset + '-' + 'train-labels-idx1-ubyte'))
            )
            test_set = (
                read_image_file(
                    os.path.join(self.root, self.raw_folder, self.dataset + '-' + 't10k-images-idx3-ubyte')),
                read_label_file(os.path.join(self.root, self.raw_folder, self.dataset + '-' + 't10k-labels-idx1-ubyte'))
            )
        elif self.dataset == 'cifar10':
            train_dict = unpickle(os.path.join(self.root, self.raw_folder, self.dataset + '_' + 'data_batch_1'))
            train_data = train_dict[b'data']
            train_label = train_dict[b'labels']
            for ii in [2, 3, 4, 5]:
                train_dict = unpickle(
                    os.path.join(self.root, self.raw_folder, self.dataset + '_' + 'data_batch_' + str(ii)))
                train_data = np.vstack((train_data, train_dict[b'data']))
                train_label.extend(train_dict[b'labels'])
            test_dict = unpickle(os.path.join(self.root, self.raw_folder, self.dataset + '_' + 'data_batch_1'))
            test_data = test_dict[b'data']
            test_label = test_dict[b'labels']
            '''
            train_data = np.rint(
                0.2989 * train_data[:, :1024] + 0.5870 * train_data[:, 1024:2048] + 0.1140 * train_data[:, 2048:])
            test_data = np.rint(
                0.2989 * test_data[:, :1024] + 0.5870 * test_data[:, 1024:2048] + 0.1140 * test_data[:, 2048:])
            train_data = train_data.reshape(50000, 32, 32)
            test_data = test_data.reshape(10000, 32, 32)
            '''
            training_set = (torch.from_numpy(train_data), torch.Tensor(train_label))
            test_set = (torch.from_numpy(test_data), torch.Tensor(test_label))
        elif self.dataset == 'SVHN':
            train_dict = scipy.io.loadmat(
                os.path.join(self.root, self.raw_folder, self.dataset + '_' + 'train_32x32.mat'))
            test_dict = scipy.io.loadmat(
                os.path.join(self.root, self.raw_folder, self.dataset + '_' + 'test_32x32.mat'))
            train_data = train_dict['X'].reshape((3072, -1))
            train_data = np.swapaxes(train_data, 0, 1)
            train_label = []
            for ii in range(train_dict['y'].shape[0]):
                if train_dict['y'][ii][0] == 10:
                    train_label.append(0)
                else:
                    train_label.append(train_dict['y'][ii][0])
            test_data = test_dict['X'].reshape((3072, -1))
            test_data = np.swapaxes(test_data, 0, 1)
            test_label = []
            for ii in range(test_dict['y'].shape[0]):
                if test_dict['y'][ii][0] == 10:
                    test_label.append(0)
                else:
                    test_label.append(test_dict['y'][ii][0])
            training_set = (torch.from_numpy(train_data), torch.Tensor(train_label))
            test_set = (torch.from_numpy(test_data), torch.Tensor(test_label))
        elif self.dataset == 'STL10':
            with open(os.path.join(self.root, self.raw_folder, self.dataset + '_' + 'train_y.bin'), 'rb') as f:
                train_label = np.fromfile(f, dtype=np.uint8) - 1
            with open(os.path.join(self.root, self.raw_folder, self.dataset + '_' + 'test_y.bin'), 'rb') as f:
                test_label = np.fromfile(f, dtype=np.uint8) - 1
            with open(os.path.join(self.root, self.raw_folder, self.dataset + '_' + 'train_X.bin'), 'rb') as f:
                everything = np.fromfile(f, dtype=np.uint8)
                images = np.reshape(everything, (-1, 3, 96, 96))
                train_data = np.transpose(images, (0, 3, 2, 1))
                train_data = np.reshape(train_data, (-1, 27648))
            with open(os.path.join(self.root, self.raw_folder, self.dataset + '_' + 'test_X.bin'), 'rb') as f:
                everything = np.fromfile(f, dtype=np.uint8)
                images = np.reshape(everything, (-1, 3, 96, 96))
                test_data = np.transpose(images, (0, 3, 2, 1))
                test_data = np.reshape(test_data, (-1, 27648))
            training_set = (torch.from_numpy(train_data), torch.from_numpy(train_label))
            test_set = (torch.from_numpy(test_data), torch.from_numpy(test_label))

        elif self.dataset == 'omniglot':
            train_d_list = []
            test_d_list = []
            train_data, test_data, train_label, test_label = [], [], [], []
            for path, dir_list, file_list in os.walk(os.path.join(self.root, self.raw_folder, self.dataset, 'train')):
                for dir_name in dir_list:
                    train_d_list.append(os.path.join(path, dir_name))
            for path, dir_list, file_list in os.walk(os.path.join(self.root, self.raw_folder, self.dataset, 'test')):
                for dir_name in dir_list:
                    test_d_list.append(os.path.join(path, dir_name))
            for ii in range(len(train_d_list)):
                for path, dir_list, file_list in os.walk(train_d_list[ii]):
                    for file_name in file_list:
                        img = plt.imread(os.path.join(train_d_list[ii], file_name)).astype(np.int8)
                        train_data.append(((1 - img) * 255).tolist())
                        train_label.append(ii)
            for ii in range(len(test_d_list)):
                for path, dir_list, file_list in os.walk(test_d_list[ii]):
                    for file_name in file_list:
                        img = plt.imread(os.path.join(test_d_list[ii], file_name)).astype(np.int8)
                        test_data.append(((1 - img) * 255).tolist())
                        test_label.append(ii)
            training_set = (torch.Tensor(train_data), torch.Tensor(train_label))
            test_set = (torch.Tensor(test_data), torch.Tensor(test_label))

        with open(os.path.join(self.root, self.processed_folder, self.dataset + self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.dataset + self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        if dataset == 'cifar10' or dataset == 'SVHN' or dataset == 'STL10':
            self.conv1 = nn.Conv2d(3, 64, 7)
        else:
            self.conv1 = nn.Conv2d(1, 64, 7)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 256, 5)
        if dataset == 'cifar10' or dataset == 'SVHN':
            self.linear1 = nn.Linear(6400, 512)
        elif dataset == 'STL10':
            self.linear1 = nn.Linear(350464, 512)
        else:
            self.linear1 = nn.Linear(2304, 512)
        self.linear2 = nn.Linear(512, 2)

    def forward(self, data):
        res = []
        for i in range(2):  # Siamese nets; sharing weights
            x = data[i]
            if dataset == 'cifar10' or dataset == 'SVHN':
                x = x.view(x.shape[0], 3, 32, 32)
            if dataset == 'STL10':
                x = x.view(x.shape[0], 3, 96, 96)
            x = self.conv1(x)
            x = F.relu(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = self.conv3(x)
            x = F.relu(x)

            x = x.view(x.shape[0], -1)
            x = self.linear1(x)
            x = F.relu(x)
            x = self.linear2(x)
            res.append(x)

        # res = torch.abs(res[1] - res[0])
        # res = self.linear2(res)
        return res[0], res[1]


def train(model, device, train_loader, epoch, optimizer):
    model.train()
    criterion = ContrastiveLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        for i in range(len(data)):
            data[i] = data[i].to(device)

        optimizer.zero_grad()
        output1, output2 = model(data[:2])
        output3, output4 = model(data[0:3:2])

        target = target.to(device)
        target_positive = torch.squeeze(target[:, 0])
        target_negative = torch.squeeze(target[:, 1])

        loss_positive = criterion(output1, output2, target_positive)
        loss_negative = criterion(output3, output4, target_negative)

        loss = loss_positive + loss_negative
        loss.backward()

        optimizer.step()
        '''
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch_size, len(train_loader.dataset),
                       100. * batch_idx * batch_size / len(train_loader.dataset),
                loss.item()))
        '''


def test(model, device, test_loader, epoch, threshold):
    model.eval()
    criterion = ContrastiveLoss()
    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        loss = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            for i in range(len(data)):
                data[i] = data[i].to(device)

            output1, output2 = model(data[:2])
            output3, output4 = model(data[0:3:2])

            target = target.to(device)
            target_positive = torch.squeeze(target[:, 0])
            target_negative = torch.squeeze(target[:, 1])

            loss_positive = criterion(output1, output2, target_positive)
            loss_negative = criterion(output3, output4, target_negative)

            loss = loss + loss_positive + loss_negative

            eucledian_distance_positive = F.pairwise_distance(output1, output2)
            eucledian_distance_negative = F.pairwise_distance(output3, output4)
            zero = torch.zeros_like(eucledian_distance_positive)
            one = torch.ones_like(eucledian_distance_positive)
            predict_positive = torch.where(eucledian_distance_positive > threshold, zero, one)
            predict_negative = torch.where(eucledian_distance_negative > threshold, zero, one)
            accurate_labels_positive = torch.sum(predict_positive).cpu()
            accurate_labels_negative = torch.sum(1 - predict_negative).cpu()

            accurate_labels = accurate_labels + accurate_labels_positive + accurate_labels_negative
            all_labels = all_labels + len(target_positive) + len(target_negative)

        accuracy = 100. * accurate_labels / all_labels
        print(
            '{}: Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(epoch + 1, accurate_labels, all_labels, accuracy,
                                                                      loss))


def oneshot(model, device, data, threshold):
    model.eval()
    result = []
    distance = []
    with torch.no_grad():
        for k in range(len(data)):
            for i in range(len(data[k])):
                data[k][i] = data[k][i].to(device)

            output1, output2 = model(data[k])
            eucledian_distance = F.pairwise_distance(output1, output2)
            distance.append(eucledian_distance.item())
            zero = torch.zeros_like(eucledian_distance)
            one = torch.ones_like(eucledian_distance)
            predict = torch.where(eucledian_distance > threshold, zero, one)
            result.append(predict.cpu().item())
    return result, distance


def main():
    setup_seed(random_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

    model = Net().to(device)

    if do_learn:  # training mode
        train_loader = torch.utils.data.DataLoader(
            BalancedMNISTPair('../data', dataset, train=True, download=True, transform=trans), batch_size=batch_size,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            BalancedMNISTPair('../data', dataset, train=False, download=True, transform=trans), batch_size=batch_size,
            shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        for epoch in range(num_epochs):
            start = time.process_time()
            train(model, device, train_loader, epoch, optimizer)
            test(model, device, test_loader, epoch, threshold)
            if (epoch + 1) % save_frequency == 0:
                # torch.save(model.state_dict(), 'siamese_cl_{:03}.pt'.format(epoch))
                torch.save(model.state_dict(), 'siamese_cnn_cl_' + dataset + '.pt'.format(epoch))
            end = time.process_time()
            print(end - start)

        prediction_loader = torch.utils.data.DataLoader(
            BalancedMNISTPair('../data', dataset, train=False, download=True, transform=trans), batch_size=1,
            shuffle=True)
        model.load_state_dict(
            torch.load('../Siamese/siamese_cnn_cl_' + dataset + '.pt'))

        data = []
        label = []
        for idx, dat in enumerate(prediction_loader):
            data.append(dat[0][:2:1])
            label.append(dat[1][0][0].detach().numpy().tolist())
            data.append(dat[0][:3:2])
            label.append(dat[1][0][1].detach().numpy().tolist())
            # data.extend(next(iter(prediction_loader))[0][:2:1])
        result = oneshot(model, device, data, threshold)
        equal = []
        for i in range(len(result)):
            equal.append((result[i] == label[i]) * 1)
        acc = sum(equal) / len(result) * 100
        print('Final test accuracy: {}/{} ({:.3f}%)'.format(sum(equal), len(result), acc))

    else:  # prediction
        prediction_loader = torch.utils.data.DataLoader(
            BalancedMNISTPair('../data', dataset, train=False, download=True, transform=trans), batch_size=1,
            shuffle=True)
        model.load_state_dict(torch.load('../Siamese/siamese_cnn_cl_' + dataset + '.pt'))

        data = []
        label = []
        equal = []
        for idx, dat in enumerate(prediction_loader):
            data.append(dat[0][:2:1])
            label.append(dat[1][0][0].detach().numpy().tolist())
            data.append(dat[0][:3:2])
            label.append(dat[1][0][1].detach().numpy().tolist())
        result, distance = oneshot(model, device, data, threshold)
        for i in range(len(result)):
            equal.append((result[i] == label[i]) * 1)
        acc = sum(equal) / len(result) * 100
        print('Final test accuracy: {}/{} ({:.3f}%)'.format(sum(equal), len(result), acc))


if __name__ == '__main__':
    main()
