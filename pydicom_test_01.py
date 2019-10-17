# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys
from os import listdir
from os.path import isfile, join
import numba
from numba import cuda

import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pydicom

MAX_VAL = 2**12

#mypath = "d:/tdcm/"
mypath = "C:/OnDemand3DApp/Users/Common/MasterDB/IMGDATA/20190228/S0000000003/"
    
try:
    dataset
except NameError:
    print('Loading dicom files')

    filenames = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    CTimind = []
    PJTimind = []
    for i, fn in enumerate(filenames, start=0):
        #print(fn)
        if pydicom.dcmread(mypath + fn).SeriesDescription == '3D CBCT Image Set':
            CTimind += [i]
        else:
            PJTimind += [i]
            
    print('Number of CT images : ' + str(len(CTimind)))
    print('Number of Thumbnail images : ' + str(len(PJTimind)))
    
    m = pydicom.dcmread(mypath + filenames[CTimind[0]]).pixel_array.shape[0]
    n = pydicom.dcmread(mypath + filenames[CTimind[0]]).pixel_array.shape[1]
    l = len(CTimind)
    dataset = np.zeros((l,m,n))
    
    for i in range(l):
        b = "Loading" + "!" * int(i*10/l) + "." * int((l-i)*10/l)
        sys.stdout.write('\r'+b)
        dataset[i] = pydicom.dcmread(mypath + filenames[CTimind[i]]).pixel_array

    print()
    print('dicom files loaded')

wx = np.zeros((3,3,3))
wx[0] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
wx[1] = [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]]
wx[2] = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

wy = np.zeros((3,3,3))
wy[0] = [[-1, -2, -1], [-2, -4, -2], [-1, -2, -1]]
wy[1] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
wy[2] = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]

wz = np.zeros((3,3,3))
wz[0] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
wz[1] = [[-2, -4, -2], [0, 0, 0], [2, 4, 2]]
wz[2] = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]


def make_plot(dataset, start = 0):
    plt.figure()
    for i, ds in enumerate(dataset[start:], start=start):
        plt.title((i, CTimind[i], pydicom.dcmread(mypath + filenames[CTimind[i]]).InstanceNumber, np.max(ds)))
        plt.imshow(ds)
        plt.pause(0.001)
        

'''
wx = np.zeros((5,5,5))
wx[0] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, -1, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
wx[1] = [[0, 0, 0, 0, 0], [0, -1, 0, 1, 0], [-1, -2, 0, 2, 1], [0, -1, 0, 1, 0], [0, 0, 0, 0, 0]]
wx[2] = [[0, -1, 0, 1, 0], [-1, -2, 0, 2, 1], [-2, -4, 0, 4, 2], [-1, -2, 0, 2, 1], [0, -1, 0, 1, 0]]
wx[3] = [[0, 0, 0, 0, 0], [0, -1, 0, 1, 0], [-1, -2, 0, 2, 1], [0, -1, 0, 1, 0], [0, 0, 0, 0, 0]]
wx[4] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, -1, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]

wy = np.zeros((5,5,5))
wy[0] = [[0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0]]
wy[1] = [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, -4, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]]
wy[2] = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
wy[3] = [[0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [1, 2, 4, 2, 1], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]]
wy[4] = [[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]

wz = np.zeros((5,5,5))
wz[0] = [[0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]
wz[1] = [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]]
wz[2] = [[0, -1, -2, -1, 0], [-1, -2, -4, -2, -1], [0, 0, 0, 0, 0], [1, 2, 4, 2, 1], [0, 1, 2, 1, 0]]
wz[3] = [[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [0, 0, 0, 0, 0], [0, 1, 2, 1, 0], [0, 0, 1, 0, 0]]
wz[4] = [[0, 0, 0, 0, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]
'''

'''
xi = 200
xf = 210
xs = 2

yi = 130
yf = 170
ys = 10

zi = 250
zf = 300
zs = 10

temp = dataset[xi:xf:xs, yi:yf:ys, zi:zf:zs]

print('deriving images')

dx = filters.convolve(temp, wx)/2**14
dy = filters.convolve(temp, wy)/2**14
dz = filters.convolve(temp, wz)/2**14

print('images derived')

x, y, z= np.meshgrid(np.arange(0, yf-yi, ys), np.arange(0, xf-xi, xs), np.arange(0, zf-zi, zs))

fig = plt.figure()
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')
ax.quiver(x, y, z, dx, dy, dz)
plt.show()
'''

print('deriving images')

def derive(dataset):
    dx = filters.convolve(dataset, wx)/2**14
    dy = filters.convolve(dataset, wy)/2**14
    dz = filters.convolve(dataset, wz)/2**14
    return dx, dy, dz

dx, dy, dz = derive(dataset)

print('images derived')

temp = dx**2 + dy**2 + dz**2


