import os
from pathlib import Path

import random
import math

import numpy
import numpy as np
import pickle as pk
import cv2
from tqdm import tqdm
from PIL import Image

import torchvision.transforms as transforms
import torch
from prefetch_generator import BackgroundGenerator
from torch.utils.data import DataLoader, Dataset, RandomSampler
from dataset.randaugment import RandAugmentMC

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return 1


# class VideoDataset(Dataset):
#
#     def __init__(self, directory_list, local_rank=0, enable_GPUs_num=0, distributed_load=False, resize_shape=[160, 160],
#                  mode='train', clip_len=32, crop_size=160):                                   # resize_shape=[224, 224]
#
#         self.clip_len, self.crop_size, self.resize_shape = clip_len, crop_size, resize_shape
#         self.mode = mode
#         self.fnames, labels = [], []
#         self.idx = []
#         # get the directory of the specified split
#         for directory in directory_list:
#             folder = Path(directory)
#             print("Load dataset from folder : ", folder)
#             for label in sorted(os.listdir(folder)):
#                 a = os.listdir(os.path.join(folder, label))
#                 for fname in os.listdir(os.path.join(folder, label)) if mode == "train" or "weak" or "strong" or "test" else os.listdir(
#                         os.path.join(folder, label))[:10]:
#                     a = fname
#                     self.fnames.append(os.path.join(folder, label, fname))
#                     labels.append(label)
#
#         random_list = list(zip(self.fnames, labels))
#         random.shuffle(random_list)
#         self.fnames[:], labels[:] = zip(*random_list)
#
#         # self.fnames = self.fnames[:240]
#         '''
#         if mode == 'train' and distributed_load:
#             single_num_ = len(self.fnames)//enable_GPUs_num
#             self.fnames = self.fnames[local_rank*single_num_:((local_rank+1)*single_num_)]
#             labels = labels[local_rank*single_num_:((local_rank+1)*single_num_)]
#         '''
#         # prepare a mapping between the label names (strings) and indices (ints)
#         self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
#         # convert the list of label names into an array of label indices
#         self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)
#
#     def __getitem__(self, index):
#         # seed = set_seed(2048)
#         # a = seed
#         # print("Random_Seed_State:", a)
#         # loading and preprocessing. TODO move them to transform classess
#         buffer = self.loadvideo(self.fnames[index])
#
#         return buffer, self.label_array[index], index
#
#
#
#     def __len__(self):
#         return len(self.fnames)
#
#     def loadvideo(self, fname):
#         # initialize a VideoCapture object to read video data into a numpy array
#         try:
#             video_stream = cv2.VideoCapture(fname)
#             frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
#         except RuntimeError:
#             index = np.random.randint(self.__len__())
#             video_stream = cv2.VideoCapture(self.fnames[index])
#             frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
#
#         while frame_count < self.clip_len + 2:
#             index = np.random.randint(self.__len__())
#             video_stream = cv2.VideoCapture(self.fnames[index])
#             frame_count = int(video_stream.get(cv2.CAP_PROP_FRAME_COUNT))
#
#         speed_rate = np.random.randint(1, 3) if frame_count > self.clip_len * 2 + 2 else 1
#         time_index = np.random.randint(frame_count - self.clip_len * speed_rate)
#
#         start_idx, end_idx, final_idx = time_index, time_index + (self.clip_len * speed_rate), frame_count - 1
#         count, sample_count, retaining = 0, 0, True
#
#         buffer = np.empty((self.clip_len, 3, self.resize_shape[0], self.resize_shape[1]), np.dtype('float16'))
#
#         self.transform = transforms.Compose([
#             transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])
#         self.transform_weak = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=1),
#             transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
#             # transforms.RandomCrop(size=224,
#             #                       padding=int(224 * 0.125),
#             #                       padding_mode='reflect'),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])
#         self.transform_strong = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=1),
#             transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
#             RandAugmentMC(n=2, m=10),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])
#         self.transform_strong_rate = transforms.Compose([
#             transforms.RandomHorizontalFlip(p=1),
#             transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])
#
#         self.transform_val = transforms.Compose([
#             # transforms.Resize([self.crop_size, self.crop_size]),
#             transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
#             transforms.ToTensor(),
#             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#         ])
#
#         if self.mode == 'train':
#             while (count <= end_idx and retaining):
#                 retaining, frame = video_stream.read()
#                 if count < start_idx:
#                     count += 1
#                     continue
#                 if count % speed_rate == speed_rate - 1 and count >= start_idx and sample_count < self.clip_len:
#                     try:
#                         buffer[sample_count] = self.transform(
#                             Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
#                     except cv2.error as err:
#                         continue
#                     sample_count += 1
#                 count += 1
#             video_stream.release()
#
#
#         elif self.mode == 'val':
#             while (count <= end_idx and retaining):
#                 retaining, frame = video_stream.read()
#                 if count < start_idx:
#                     count += 1
#                     continue
#                 if count % speed_rate == speed_rate - 1 and count >= start_idx and sample_count < self.clip_len:
#                     try:
#                         buffer[sample_count] = self.transform_val(
#                             Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
#                     except cv2.error as err:
#                         continue
#                     sample_count += 1
#                 count += 1
#             video_stream.release()
#
#         elif self.mode == 'weak':
#             while (count <= end_idx and retaining):
#                 retaining, frame = video_stream.read()
#                 if count < start_idx:
#                     count += 1
#                     continue
#                 if count % speed_rate == speed_rate - 1 and count >= start_idx and sample_count < self.clip_len:
#                     try:
#                         buffer[sample_count] = self.transform_weak(
#                             Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
#                     except cv2.error as err:
#                         continue
#                     sample_count += 1
#                 count += 1
#             video_stream.release()
#
#         elif self.mode == 'strong':
#             while (count <= end_idx and retaining):
#                 retaining, frame = video_stream.read()
#                 if count < start_idx:
#                     count += 1
#                     continue
#                 if count % speed_rate == speed_rate - 1 and count >= start_idx and sample_count < self.clip_len:
#                     try:
#                         buffer[sample_count] = self.transform_strong(
#                             Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
#                     except cv2.error as err:
#                         continue
#                     sample_count += 1
#                 count += 1
#             video_stream.release()
#
#         elif self.mode == 'test':
#             while (count <= end_idx and retaining):
#                 retaining, frame = video_stream.read()
#                 if count < start_idx:
#                     count += 1
#                     continue
#                 if count % speed_rate == speed_rate - 1 and count >= start_idx and sample_count < self.clip_len:
#                     try:
#                         buffer[sample_count] = self.transform_val(
#                             Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
#                     except cv2.error as err:
#                         continue
#                     sample_count += 1
#                 count += 1
#             video_stream.release()
#
#
#         return buffer.transpose((1, 0, 2, 3))

#
# #===================================== PKL ====================================
class VideoDataset(Dataset):

    def __init__(self, directory_list, local_rank=0, enable_GPUs_num=0, distributed_load=False, resize_shape=[160, 160],
                 mode='train', clip_len=32, crop_size=160):                                   # resize_shape=[224, 224]

        self.clip_len, self.crop_size, self.resize_shape = clip_len, crop_size, resize_shape
        self.mode = mode
        self.fnames, labels = [], []
        self.idx = []
        # get the directory of the specified split
        for directory in directory_list:
            folder = Path(directory)
            print("Load dataset from folder : ", folder)
            for label in sorted(os.listdir(folder)):
                a = os.listdir(os.path.join(folder, label))
                for fname in os.listdir(os.path.join(folder, label)) if mode == "train" or "weak" or "strong" or "test" else os.listdir(
                        os.path.join(folder, label))[:10]:
                    a = fname
                    self.fnames.append(os.path.join(folder, label, fname))
                    labels.append(label)

        random_list = list(zip(self.fnames, labels))
        random.shuffle(random_list)
        self.fnames[:], labels[:] = zip(*random_list)

        # self.fnames = self.fnames[:240]
        '''
        if mode == 'train' and distributed_load:
            single_num_ = len(self.fnames)//enable_GPUs_num
            self.fnames = self.fnames[local_rank*single_num_:((local_rank+1)*single_num_)]
            labels = labels[local_rank*single_num_:((local_rank+1)*single_num_)]
        '''
        # prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

    def __getitem__(self, index):
        # seed = set_seed(2048)
        # a = seed
        # print("Random_Seed_State:", a)
        # loading and preprocessing. TODO move them to transform classess
        buffer = self.loadvideo(self.fnames[index])

        return buffer, self.label_array[index], index


    def __len__(self):
        return len(self.fnames)

    def loadvideo(self, fname):
        # seed = set_seed(2048)
        # print("Random_Seed_State:",seed)
        # initialize a VideoCapture object to read video data into a numpy array
        with open(fname, 'rb') as Video_reader:
            try:
                video = pk.load(Video_reader)
            except EOFError:
                return None



        while video.shape[0] < self.clip_len + 2:
            index = np.random.randint(self.__len__())
            with open(self.fnames[index], 'rb') as Video_reader:
                video = pk.load(Video_reader)

        height, width = video.shape[1], video.shape[2]

        speed_rate = np.random.randint(1, 3) if video.shape[0] > self.clip_len * 2 + 2 and self.mode == "train" else 1
        time_index = np.random.randint(video.shape[0] - self.clip_len * speed_rate)

        video = video[time_index:time_index + (self.clip_len * speed_rate):speed_rate, :, :, :]

        self.transform = transforms.Compose([
            transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.transform_weak = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
            # transforms.RandomCrop(size=224,
            #                       padding=int(224 * 0.125),
            #                       padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.transform_strong = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
            # transforms.RandomCrop(size=224,
            #                       padding=int(224 * 0.125),
            #                       padding_mode='reflect'),
            RandAugmentMC(n=2, m=10),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.transform_strong_rate = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1),
            transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.transform_val = transforms.Compose([
            # transforms.Resize([self.crop_size, self.crop_size]),
            transforms.Resize([self.resize_shape[0], self.resize_shape[1]]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        if self.mode == 'train':
            # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
            buffer = np.empty((self.clip_len, 3, self.resize_shape[0], self.resize_shape[1]), np.dtype('float16'))
            for idx, frame in enumerate(video):
                buffer[idx] = self.transform(Image.fromarray(frame))

        elif self.mode == 'val':
            # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
            buffer = np.empty((self.clip_len, 3, self.resize_shape[0], self.resize_shape[1]), np.dtype('float16'))
            for idx, frame in enumerate(video):
                buffer[idx] = self.transform_val(Image.fromarray(frame))

        elif self.mode == 'weak':
            buffer = np.empty((self.clip_len, 3, self.resize_shape[0], self.resize_shape[1]), np.dtype('float16'))
            for idx, frame in enumerate(video):
                buffer[idx] = self.transform_weak(Image.fromarray(frame))

        elif self.mode == 'strong':
            buffer = np.empty((self.clip_len, 3, self.resize_shape[0], self.resize_shape[1]), np.dtype('float16'))
            for idx, frame in enumerate(video):
                buffer[idx] = self.transform_strong(Image.fromarray(frame))


                    # if idx % 3 == 0:
                    #     buffer[idx] = self.transform_strong(Image.fromarray(frame))
                    # else:
                    #     buffer[idx] = self.transform_strong_rate(Image.fromarray(frame))


        elif self.mode == 'test':
            buffer = np.empty((self.clip_len, 3, self.resize_shape[0], self.resize_shape[1]), np.dtype('float16'))
            for idx, frame in enumerate(video):
                buffer[idx] = self.transform_val(Image.fromarray(frame))



        return buffer.transpose((1, 0, 2, 3))

class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


def Get_Dataloader(datapath, mode, bs):
    dataset = VideoDataset(datapath,
                           mode=mode)
    Label_dict = dataset.label2index

    dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, num_workers=8)


    return dataloader, list(Label_dict.keys())

def Get_lx_sux_wux_Dataloader_forSI9(args, datapath, weak_datapath, strong_datapath, mode, bs):
    dataset = VideoDataset(datapath,
                           mode=mode)

    weak_dataset = VideoDataset(weak_datapath,
                                mode='weak')
    strong_dataset = VideoDataset(strong_datapath,
                                  mode='strong')

    train_labeled_idxs, train_unlabeled_idxs = x_u_split_SI9(
        args, dataset.label_array)

    # train_unlabeled_idxs, _ = x_u_split(
    #     args, dataset.label_array)

    labeled_train_dataset = get_ucf101_ssl(dataset, train_labeled_idxs)
    print("-------------------------------------------")
    dataset = VideoDataset(datapath,
                           mode=mode)
    unlabeled_train_dataset = get_ucf101_ssl(dataset, train_unlabeled_idxs)
    print("-------------------------------------------")
    # dataset = VideoDataset(datapath,
    #                        resize_shape=[224, 224],
    #                        mode=mode)
    # unlabeled_train_dataset = get_ucf101_ssl(dataset, train_unlabeled_idxs)
    unlabeled_weak_dataset = get_ucf101_ssl(weak_dataset, train_unlabeled_idxs)
    print("-------------------------------------------")
    unlabeled_strong_dataset = get_ucf101_ssl(strong_dataset, train_unlabeled_idxs)
    print("------------------------------------------")

    u_bs = int(bs * args.mu)
    random_sampler = RandomSampler(unlabeled_train_dataset)
    labeled_train_dataloader = DataLoader(labeled_train_dataset,
                                          batch_size=bs,
                                          shuffle=True,
                                          num_workers=8)

    unlabeled_train_dataloader = DataLoader(unlabeled_train_dataset,
                                            batch_size=u_bs,
                                            sampler=random_sampler,
                                            shuffle=False,
                                            num_workers=8)

    unlabeled_weak_dataloader = DataLoader(unlabeled_weak_dataset,
                                           batch_size=u_bs,
                                           sampler=random_sampler,
                                           shuffle=False,
                                           num_workers=8)

    unlabeled_strong_dataloader = DataLoader(unlabeled_strong_dataset,
                                             batch_size=u_bs,
                                             sampler=random_sampler,
                                             shuffle=False,
                                             num_workers=8)

    return labeled_train_dataloader, unlabeled_train_dataloader, unlabeled_weak_dataloader, unlabeled_strong_dataloader


def Get_lx_sux_wux_Datasets_forSI9(args, datapath, weak_datapath, strong_datapath, mode, bs):
    dataset = VideoDataset(datapath,
                           mode=mode)

    weak_dataset = VideoDataset(weak_datapath,
                                mode='weak')
    strong_dataset = VideoDataset(strong_datapath,
                                  mode='strong')

    train_labeled_idxs, train_unlabeled_idxs = x_u_split_SI9(
        args, dataset.label_array)

    # train_unlabeled_idxs, _ = x_u_split(
    #     args, dataset.label_array)

    labeled_train_dataset = get_ucf101_ssl(dataset, train_labeled_idxs)
    print("-------------------------------------------")
    dataset = VideoDataset(datapath,
                           mode=mode)
    unlabeled_train_dataset = get_ucf101_ssl(dataset, train_unlabeled_idxs)
    print("-------------------------------------------")
    # dataset = VideoDataset(datapath,
    #                        resize_shape=[224, 224],
    #                        mode=mode)
    # unlabeled_train_dataset = get_ucf101_ssl(dataset, train_unlabeled_idxs)
    unlabeled_weak_dataset = get_ucf101_ssl(weak_dataset, train_unlabeled_idxs)
    print("-------------------------------------------")
    unlabeled_strong_dataset = get_ucf101_ssl(strong_dataset, train_unlabeled_idxs)
    print("------------------------------------------")



    return labeled_train_dataset, unlabeled_train_dataset, unlabeled_weak_dataset, unlabeled_strong_dataset

def adjust_label_per_class(label_per_class, num_labeled):
   
    a = sum(label_per_class)

    
    b = a - num_labeled

    if b > 0:
        
        max_value = max(label_per_class)
        max_index = label_per_class.index(max_value)
        label_per_class[max_index] -= b
    elif b < 0:
        
        min_value = min(label_per_class)
        min_index = label_per_class.index(min_value)
        label_per_class[min_index] += abs(b)

    return label_per_class

def x_u_split_SI9(args, labels):

    label_per_class = []
    for i in range(args.num_classes):
        try:
            count = np.count_nonzero(labels == i)
            len_of_dataset = len(labels)
            class_num = math.ceil(  count  * (args.num_labeled/len_of_dataset) )
            label_per_class.append(class_num)
        except:
            print("split labels error")

    # label_per_class = args.num_labeled // args.num_classes   # 12 / 2 = 6
    label_per_class = adjust_label_per_class(label_per_class, args.num_labeled)
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        try:
            idx = np.where(labels == i)[0]
            a = len(idx)
            idx = np.random.choice(idx, label_per_class[i], False)
            labeled_idx.extend(idx)
        except:
            print("error idx length = ", a, '\n')
            print("error class = ", i)

        # idx = np.where(labels == i)[0]
        # a = len(idx)
        # idx = np.random.choice(idx, label_per_class, False)
        # labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    same = []
    # same = np.array(same)
    for i in range(0,len(labeled_idx)):
        for j in range(0,len(unlabeled_idx)):
            if labeled_idx[i] == unlabeled_idx[j]:
                same.append(unlabeled_idx[j])

    unlabeled_idx = numpy.delete(unlabeled_idx, same)

    return labeled_idx, unlabeled_idx

def x_u_split(args, labels):
    label_per_class = args.num_labeled // args.num_classes   # 12 / 2 = 6
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        try:
            idx = np.where(labels == i)[0]
            a = len(idx)
            idx = np.random.choice(idx, label_per_class, False)
            labeled_idx.extend(idx)
        except:
            print("error idx length = ", a, '\n')
            print("error class = ", i)

        # idx = np.where(labels == i)[0]
        # a = len(idx)
        # idx = np.random.choice(idx, label_per_class, False)
        # labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    a = args.num_labeled
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    same = []
    # same = np.array(same)
    for i in range(0,len(labeled_idx)):
        for j in range(0,len(unlabeled_idx)):
            if labeled_idx[i] == unlabeled_idx[j]:
                same.append(unlabeled_idx[j])

    unlabeled_idx = numpy.delete(unlabeled_idx, same)

    return labeled_idx, unlabeled_idx


def back_index(args, labels):
    # label_per_class = args.num_labeled // args.num_classes   # 12 / 2 = 6
    labels = np.array(labels)
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    for i in range(args.num_classes):
        try:
            idx = np.where(labels == i)[0]
            a = len(idx)
            # idx = np.random.choice(idx, label_per_class, False)
            labeled_idx.extend(idx)
        except:
            print("error idx length = ", a, '\n')
            print("error class = ", i)

        # idx = np.where(labels == i)[0]
        # a = len(idx)
        # idx = np.random.choice(idx, label_per_class, False)
        # labeled_idx.extend(idx)
    labeled_idx = np.array(labeled_idx)
    a = args.num_labeled
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    same = []
    # same = np.array(same)
    for i in range(0,len(labeled_idx)):
        for j in range(0,len(unlabeled_idx)):
            if labeled_idx[i] == unlabeled_idx[j]:
                same.append(unlabeled_idx[j])

    unlabeled_idx = numpy.delete(unlabeled_idx, same)

    return labeled_idx, unlabeled_idx

def get_ucf101_ssl(dataset,indexs):
    ssl_dataset = dataset
    ilen = len(indexs)
    data = []
    target = []
    

    for i in range(0,(len(indexs))):
        if i == 0:
            aa = 0

        # aa = ssl_dataset.fnames[indexs[i]]
        # bb = ssl_dataset.label_array[indexs[i]] + 1


        #ssl_dataset.fnames 0-13319
        #ssl_dataset.label_array 0-13319
        # try:
        a = indexs[i]
        b = ssl_dataset.fnames.__len__()
        data.append(ssl_dataset.fnames[indexs[i]])
        # target.append(ssl_dataset.label_array[indexs[i]]+1)
        target.append(ssl_dataset.label_array[indexs[i]])
        # except:
        #     print()



    ssl_dataset.fnames = data
    ssl_dataset.label_array = np.array(target)
    return ssl_dataset





