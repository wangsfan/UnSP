# -*- coding: utf-8 -*-
"""
@File  : my_utils.py
@Author: 林嚣张
@Date  : 2023/4/24 17:08
@Software  : pycharm
"""
import math
import time
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.animation import ArtistAnimation


class testset:
    def __init__(self, root_dir, ev_rate, norm_e, e_abs, num_freams):

        self.num_frm = num_freams
        self.norm_e = norm_e
        self.e_abs = e_abs
        self.bs = 1
        self.h = 180
        self.w = 240
        self.img_num = 0
        self.bins = 5
        self.root_dir = root_dir + '/'
        self.events_file = 'events.txt'
        self.img_file = 'images.txt'
        self.num_events = int(ev_rate * self.h * self.w)
        self.args = [self.bins, self.w, self.h]  # num_bins, width, height
        self.iterator = pd.read_csv(self.root_dir + self.events_file, header=None,
                                    delimiter=' ',
                                    names=['t', 'x', 'y', 'p'],
                                    dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'p': np.int16},
                                    engine='c',
                                    index_col=False)

        self.img_metadata = pd.read_csv(self.root_dir + self.img_file,
                                        delimiter=' ',
                                        header=None, names=['t', 'fname'],
                                        index_col=False)
        self.time_stamps = pd.read_csv(self.root_dir + 'time_stamp.csv')

        max_time_stamp = self.time_stamps.iloc[num_freams + 1][1]
        self.iterator = self.iterator[:max_time_stamp + 1]

    def getitem(self, item):

        first_time_stamp = self.time_stamps.iloc[item][1]
        new_time_stamp = self.time_stamps.iloc[item + 1][1]
        event_tensor = self.iterator.values[first_time_stamp:new_time_stamp]
        num_evs = event_tensor.shape[0] // self.num_events
        if num_evs == 0:
            num_evs = 1
        event_tensor = np.array_split(event_tensor, num_evs, axis=0)
        evs = torch.zeros(num_evs, 5, self.h, self.w)
        img_name = self.root_dir + self.img_metadata.fname[item]

        with torch.no_grad():
            for i in range(num_evs):
                ev_ten = torch.from_numpy(event_tensor[i])
                if self.e_abs:
                    ev_ten[:, 3][ev_ten[:, 3] == -1] = 1
                    ev_ten[:, 3][ev_ten[:, 3] == 0] = 1
                evs[i] = self.events_to_voxel_grid_pytorch(ev_ten, *self.args)
                if self.norm_e:
                    evs[i] = self.norm(evs[i])
            img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE) / 255.0
            img = torch.from_numpy(img).float()

        return evs, img

    def norm(self, events):
        with torch.no_grad():
            nonzero_ev = (events != 0)
            num_nonzeros = nonzero_ev.sum()
            if num_nonzeros > 0:
                mean = events.sum() / num_nonzeros
                stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
                mask = nonzero_ev.float()
                events = mask * (events - mean) / (stddev + 1e-8)
        return events



class Timer:
    def __init__(self, msg='Time elapsed'):
        self.msg = msg

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        duration = self.end - self.start
        print(f'{self.msg}: {duration:.2f}s')


class Event:
    __slots__ = 't', 'x', 'y', 'p'

    def __init__(self, t, x, y, p):
        self.t = t
        self.x = x
        self.y = y
        self.p = p

    def __repr__(self):
        return f'Event(t={self.t:.3f}, x={self.x}, y={self.y}, p={self.p})'


def normalize_image(image, percentile_lower=1, percentile_upper=99):
    mini, maxi = np.percentile(image, (percentile_lower, percentile_upper))
    if mini == maxi:
        return 0 * image + 0.5  # gray image
    return np.clip((image - mini) / (maxi - mini + 1e-5), 0, 1)


class EventData:
    def __init__(self, event_list, width, height):
        self.event_list = event_list
        self.width = width
        self.height = height

    def add_frame_data(self, data_folder, max_frames=100):
        timestamps = np.genfromtxt(join(data_folder, 'image_timestamps.txt'), max_rows=int(max_frames))
        frames = []
        frame_timestamps = []
        with open(join(data_folder, 'image_timestamps.txt')) as f:
            for line in f:
                fname, timestamp = line.split(' ')
                timestamp = float(timestamp)
                frame = cv2.imread(join(data_folder, fname), cv2.IMREAD_GRAYSCALE)
                if not (frame.shape[0] == self.height and frame.shape[1] == self.width):
                    continue
                frames.append(frame)
                frame_timestamps.append(timestamp)
                if timestamp >= self.event_list[-1].t:
                    break
        frame_stack = normalize_image(np.stack(frames, axis=0))
        self.frames = [f for f in frame_stack]
        self.frame_timestamps = frame_timestamps


def animate(images, fig_title=''):
    fig = plt.figure(figsize=(0.1, 0.1))  # don't take up room initially
    fig.suptitle(fig_title)
    fig.set_size_inches(7.2, 5.4, forward=False)  # resize but don't update gui
    ims = []
    for image in images:
        im = plt.imshow(normalize_image(image), cmap='gray', vmin=0, vmax=1, animated=True)
        ims.append([im])
    ani = ArtistAnimation(fig, ims, interval=50, blit=False, repeat_delay=1000)
    plt.close(ani._fig)
    return ani


def load_events(path_to_events, n_events=None):
    print('Loading events...')
    header = pd.read_csv(path_to_events, delim_whitespace=True, names=['width', 'height'],
                         dtype={'width': np.int, 'height': np.int}, nrows=1)
    width, height = 240, 180
    print(f'width, height: {width}, {height}')
    event_pd = pd.read_csv(path_to_events, delim_whitespace=True, header=None,
                           names=['t', 'x', 'y', 'p'],
                           dtype={'t': np.float64, 'x': np.int16, 'y': np.int16, 'p': np.int8},
                           engine='c', skiprows=1, nrows=n_events, memory_map=True)
    event_list = []
    for event in event_pd.values:
        t, x, y, p = event
        event_list.append(Event(t, int(x), int(y), -1 if p < 0.5 else 1))
    print('Loaded {:.2f}M events'.format(len(event_list) / 1e6))
    return EventData(event_list, width, height)


def plot_3d(event_data, n_events=-1):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    x, y, t, c = [], [], [], []
    for e in event_data.event_list[:int(n_events)]:
        x.append(e.x)
        y.append(e.y)
        t.append(e.t * 1e3)
        c.append('r' if e.p == 1 else 'b')
    ax.scatter(t, x, y, c=c, marker='.')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_zlim(*ax.get_zlim()[::-1])  # reverse 'y' image axis
    plt.show()


def event_slice(event_data, start=0, duration_ms=30):
    events, height, width = event_data.event_list, event_data.height, event_data.width
    mask = np.zeros((height, width), dtype=np.int8)
    start_idx = int(start * (len(events) - 1))
    end_time = events[start_idx].t + duration_ms / 1000.0
    for e in events[start_idx:]:
        mask[e.y, e.x] = e.p
        if e.t >= end_time:
            break
    img_rgb = np.ones((height, width, 3), dtype=np.uint8) * 255
    img_rgb[mask == -1] = (255, 0, 0)
    img_rgb[mask == 1] = (0, 0, 255)
    fig = plt.figure(figsize=(7.2, 5.4))
    plt.imshow(img_rgb)


def high_pass_filter(event_data, cutoff_frequency=5):
    print('Reconstructing, please wait...')
    events, height, width = event_data.event_list, event_data.height, event_data.width
    events_per_frame = 2e4
    with Timer('Reconstruction'):
        time_surface = np.zeros((height, width), dtype=np.float32)
        image_state = np.zeros((height, width), dtype=np.float32)
        image_list = []
        for i, e in enumerate(events):
            beta = math.exp(-cutoff_frequency * (e.t - time_surface[e.y, e.x]))
            image_state[e.y, e.x] = beta * image_state[e.y, e.x] + e.p
            time_surface[e.y, e.x] = e.t
            if i % events_per_frame == 0:
                beta = np.exp(-cutoff_frequency * (e.t - time_surface))
                image_state *= beta
                time_surface.fill(e.t)
                image_list.append(np.copy(image_state))
    return image_list


def leaky_integrator(event_data, beta=1.0):
    print('Reconstructing, please wait...')
    events, height, width = event_data.event_list, event_data.height, event_data.width
    events_per_frame = 2e4
    with Timer('Reconstruction (simple)'):
        image_state = np.zeros((height, width), dtype=np.float32)
        image_list = []
        for i, e in enumerate(events):
            image_state[e.y, e.x] = beta * image_state[e.y, e.x] + e.p
            if i % events_per_frame == 0:
                image_list.append(np.copy(image_state))
    fig_title = 'Direct Integration' if beta == 1 else 'Leaky Integrator'
    return animate(image_list, fig_title)


def complementary_filter(event_data, cutoff_frequency=5.0):
    print('Reconstructing, please wait...')
    events, height, width = event_data.event_list, event_data.height, event_data.width
    frames, frame_timestamps = event_data.frames, event_data.frame_timestamps
    events_per_frame = 2e4
    with Timer('Reconstruction'):
        time_surface = np.zeros((height, width), dtype=np.float32)
        image_state = np.zeros((height, width), dtype=np.float32)
        image_list = []
        frame_idx = 0
        max_frame_idx = len(frames) - 1
        log_frame = np.log(frames[0] + 1)
        for i, e in enumerate(events):
            if frame_idx < max_frame_idx:
                if e.t >= frame_timestamps[frame_idx + 1]:
                    log_frame = np.log(frames[frame_idx + 1] + 1)
                    frame_idx += 1
            beta = math.exp(-cutoff_frequency * (e.t - time_surface[e.y, e.x]))
            image_state[e.y, e.x] = beta * image_state[e.y, e.x] \
                                    + (1 - beta) * log_frame[e.y, e.x] + 0.1 * e.p
            time_surface[e.y, e.x] = e.t
            if i % events_per_frame == 0:
                beta = np.exp(-cutoff_frequency * (e.t - time_surface))
                image_state = beta * image_state + (1 - beta) * log_frame
                time_surface.fill(e.t)
                image_list.append(np.copy(image_state))
    return animate(image_list, 'Complementary Filter')


def main():
    with Timer('Loading'):
        n_events = 5e5
        path_to_events = '/media/thc/Elements/unsp/dvs_datasets/dynamic_6dof/events.txt'
        event_data = load_events(path_to_events, n_events)
    plot_3d(event_data=event_data, n_events=5000)
    img_list = high_pass_filter(event_data=event_data, cutoff_frequency=5)
    print(len(img_list))


if __name__ == '__main__':
    main()
