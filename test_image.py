import math
import numpy as np
import pandas as pd
import cv2
import time
import filter

import matplotlib.pyplot as plt
from IPython.display import display


class Node(object):
    def __init__(self, x, y, value):
        self.x = x
        self.y = y
        self.value = value


def getBlackChannel(img):
    if len(img.shape) == 3 and img.shape[2] == 3:
        pass
    else:
        print("Необходимо цветное изображение")
        return None

    return np.min(img, axis=2)


def getDarkChannel(img, block_size=3):
    if len(img.shape) == 2:
        pass
    else:
        print("Размер != 2")
        return None

    if block_size % 2 == 0 or block_size < 3:
        print('Размер нечетный или слишком мал')
        return None

    A = int((block_size - 1) / 2)

    H = img.shape[0] + block_size - 1
    W = img.shape[1] + block_size - 1

    imgMiddle = 255 * np.ones((H, W))

    imgMiddle[A:H - A, A:W - A] = img

    imgDark = np.zeros_like(img, np.uint8)

    localMin = 255
    for i in range(A, H - A):
        for j in range(A, W - A):
            x = range(i - A, i + A + 1)
            y = range(j - A, j + A + 1)
            imgDark[i - A, j - A] = np.min(imgMiddle[x, y])

    return imgDark


def getAtomsphericLight(dark_channel, img, mean_mode=False, percent=0.001):
    size = dark_channel.shape[0] * dark_channel.shape[1]
    height = dark_channel.shape[0]
    width = dark_channel.shape[1]

    nodes = []

    for i in range(0, height):
        for j in range(0, width):
            oneNode = Node(i, j, dark_channel[i, j])
            nodes.append(oneNode)

    nodes = sorted(nodes, key=lambda node: node.value, reverse=True)

    atmospheric_light = 0

    if int(percent * size) == 0:
        for i in range(0, 3):
            if img[nodes[0].x, nodes[0].y, i] > atmospheric_light:
                atmospheric_light = img[nodes[0].x, nodes[0].y, i]
        return atmospheric_light

    if mean_mode:
        sum_node = 0
        for i in range(0, int(percent * size)):
            for j in range(0, 3):
                sum_node = sum_node + img[nodes[i].x, nodes[i].y, j]
        atmospheric_light = int(sum_node / (int(percent * size) * 3))
        return atmospheric_light

    for i in range(0, int(percent * size)):
        for j in range(0, 3):
            if img[nodes[i].x, nodes[i].y, j] > atmospheric_light:
                atmospheric_light = img[nodes[i].x, nodes[i].y, j]
    return atmospheric_light


def getRecoverScene(img, omega=0.98, t0=0.1, block_size=9, mean_mode=False, percent=0.003, refine=True):
    imgGray = getBlackChannel(img)
    imgDark = getDarkChannel(imgGray, block_size=block_size)
    atmosphericLight = getAtomsphericLight(imgDark, img, mean_mode=mean_mode, percent=percent)

    imgDark = np.float64(imgDark)
    transmission = 1 - omega * imgDark / atmosphericLight

    transmission[transmission < 0.1] = 0.1

    if refine:
        normal = (img - img.min()) / (img.max() - img.min())  # normalize I
        transmission = filter.real_filter(normal, transmission, r=40, eps=1e-3)

    scene_radiance = np.zeros(img.shape)
    img = np.float64(img)

    for i in range(3):
        SR = (img[:, :, i] - atmosphericLight) / transmission + atmosphericLight

        SR[SR > 255] = 255
        SR[SR < 0] = 0
        scene_radiance[:, :, i] = SR

    scene_radiance = np.uint8(scene_radiance)

    return scene_radiance
