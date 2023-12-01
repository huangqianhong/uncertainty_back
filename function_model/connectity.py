"""
连通性
"""
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import label
from numpy.random import default_rng
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.cluster import KMeans
import pywt
from urllib.request import urlopen
import gzip
import pandas as pd
import glob, sys

from common.file_util import read_file_to_arr_2d

base = np.e


def connect_file(ti_path, realization_path):
    realization_arr, x, y = read_file_to_arr_2d(realization_path)
    ti_arr, ti_x, ti_y = read_file_to_arr_2d(ti_path)
    yyy, xxx = np.meshgrid(np.arange(1, ti_y + 1), np.arange(1, ti_x + 1))
    # xx = xxx[slice_iz, :, :]
    # yy = yyy[slice_iz, :, :]
    # zz = zzz[slice_iz, :, :]
    xx = xxx[:, :]
    yy = yyy[:, :]
    zz = xxx[:, :]  # 没有zz
    maxnbsamples = int(0.3E3)
    pnorm = 2
    nblags = 12
    maxh = np.sqrt(ti_x ** 2 + ti_y ** 2) / 3
    connectivity_propability, connectivity_distance_list, connectivity_lagcp_list = dist_lpnorm_categorical_lag_connectivity_add_disance_lagpc(
        ti_arr, realization_arr, xx, yy, zz,
        nblags, maxh,
        maxnbsamples, pnorm,
        clblab="", plot=True, verb=True)
    # todo 连通性这边主要用于2维的，后面可以做连接 connectivity_distance_list, connectivity_lagcp_list 主要做连通性变差图

    return connectivity_propability, connectivity_distance_list, connectivity_lagcp_list


# 添加多个参数，方便前端进行折现图的渲染
def dist_lpnorm_categorical_lag_connectivity_add_disance_lagpc(img1, img2, xxx, yyy, zzz, nblags, maxh, maxnbsamples,
                                                               pnorm, clblab='', plot=False, verb=False, slice_ix=0,
                                                               slice_iy=0, slice_iz=0):
    d = 0

    # identify all indicators
    indicators = np.unique(np.hstack((img1.flatten(), img2.flatten())))
    nbind = len(indicators)
    d_ind = np.zeros(nbind)
    # for all indicators
    for i in range(nbind):
        classcode = indicators[i]
        if verb:
            print('indicator ' + str(i))
        img1bin = ((img1 == classcode) * 1).astype(int)
        img2bin = ((img2 == classcode) * 1).astype(int)
        img1cnt = np.sum(img1bin)
        img2cnt = np.sum(img2bin)
        if img1cnt + img2cnt == 0:
            d_ind[i] = 0
        elif img1cnt * img2cnt == 0:
            d_ind[i] = 1 / nbind
        else:
            if verb:
                print('img1 compute indicator_lag_connectivity')
            [lag_xc1, lag_ct1, lag_cp1] = indicator_lag_connectivity(img1bin, xxx, yyy, zzz, nblags, maxh, maxnbsamples,
                                                                     verb=verb)
            if verb:
                print('img2 compute indicator_lag_connectivity')
            [lag_xc2, lag_ct2, lag_cp2] = indicator_lag_connectivity(img2bin, xxx, yyy, zzz, nblags, maxh, maxnbsamples,
                                                                     verb=verb)
            d_ind[i] = weighted_lpnorm(lag_cp1, lag_cp2, pnorm, verb=verb)
        d += 1 / nbind * d_ind[i] ** pnorm
        if verb:
            print('distance contribution: ' + str(d_ind[i]))
        if plot:
            plot_ind_cty(img1, img2, lag_xc1, lag_cp1, lag_xc2, lag_cp2, classcode, clblab=clblab, slice_ix=slice_ix,
                         slice_iy=slice_iy, slice_iz=slice_iz)
    d = d ** (1 / pnorm)
    lag_xc1 = lag_xc1.astype(int)
    lagc_list = lag_xc1.tolist()
    lagcp_list = [lag_cp1.tolist(), lag_cp2.tolist()]

    return d, lagc_list, lagcp_list


def indicator_lag_connectivity(array, xxx, yyy, zzz, nblags, maxh, maxnbsamples, clblab='', verb=False):
    lag_count = np.zeros(nblags) + np.nan  # lag center
    lag_proba = np.zeros(nblags) + np.nan  # connectivity probability
    lag_center = (np.arange(nblags) + 1) * maxh / nblags  # count per lag
    if np.sum(array) == 0:
        return lag_center, lag_count, lag_proba
    array_size = np.prod(array.shape)
    laglim = np.linspace(0, maxh, nblags + 1)
    clblabed_array, num_features = label(array)  # clblab array
    clblabed_array = np.reshape(clblabed_array, (array_size, 1)).flatten()
    ix_c = (np.asarray(np.where(clblabed_array > 0))).flatten()
    ix_rn = (np.round(
        np.random.uniform(0, 1, int(min(maxnbsamples, np.sum(array), len(ix_c)))) * (np.sum(array) - 1))).astype(int)
    samples_ix = ix_c[ix_rn]
    samples_val = clblabed_array[samples_ix]
    samples_xxx = np.reshape(xxx, (array_size, 1)).flatten()[samples_ix]
    samples_yyy = np.reshape(yyy, (array_size, 1)).flatten()[samples_ix]
    samples_zzz = np.reshape(zzz, (array_size, 1)).flatten()[samples_ix]
    # compute distance and square diff between sampled pair of points
    dist = np.zeros(np.round(len(samples_ix) * (len(samples_ix) - 1) / 2).astype(int)) + np.nan
    conn = np.zeros(np.round(len(samples_ix) * (len(samples_ix) - 1) / 2).astype(int)) + np.nan
    k = 0
    if verb:
        print('computing distance and connexion for each sampled pair of point')
    for i in range(len(samples_ix)):
        for j in np.arange(i):
            dist[k] = ((samples_xxx[i] - samples_xxx[j]) ** 2 + (samples_yyy[i] - samples_yyy[j]) ** 2 + (
                    samples_zzz[i] - samples_zzz[j]) ** 2) ** 0.5
            conn[k] = 1 - (samples_val[i] != samples_val[j]) ** 2
            k += 1
    # for each lag
    if verb:
        print('computing connexion probability per lag')
    for l in range(nblags):
        # identify sampled pairs belonging to the lag
        lag_lb = laglim[l]
        lag_ub = laglim[l + 1]
        ix = np.where((dist >= lag_lb) & (dist < lag_ub))
        # count, experimental semi vario value and center of lag cloud
        lag_count[l] = len(ix[0])
        if len(ix[0]) > 0:
            lag_center[l] = np.mean(dist[ix])
            lag_proba[l] = np.mean(conn[ix])
    return lag_center, lag_count, lag_proba


def plot_ind_cty(img1, img2, lag_xc1, lag_cp1, lag_xc2, lag_cp2, classcode, clblab="", slice_ix=0, slice_iy=0,
                 slice_iz=0):
    ndim = len(img1.shape)
    vmin = np.min([np.min(img1), np.min(img2)])
    vmax = np.max([np.max(img1), np.max(img2)])
    if ndim == 3:
        fig = plt.figure()
        gs = fig.add_gridspec(2, 7)
        ax00 = fig.add_subplot(gs[0:, 3])
        ax01 = fig.add_subplot(gs[0, 0])
        ax02 = fig.add_subplot(gs[0, 1])
        ax03 = fig.add_subplot(gs[0, 2])
        ax11 = fig.add_subplot(gs[1, 0])
        ax12 = fig.add_subplot(gs[1, 1])
        ax13 = fig.add_subplot(gs[1, 2])
        ax4 = fig.add_subplot(gs[0:, 4:])
        axins = inset_axes(ax00,
                           width="10%",  # width = 5% of parent_bbox width
                           height="90%",  # height : 50%
                           loc='center left'
                           )
        ax00.axis('off')
        ax01.axis('off')
        ax02.axis('off')
        ax03.axis('off')
        ax01.set_title('img1 Map')
        ax02.set_title('img1 W (N) E')
        ax03.set_title('img1 N (W) S')
        ax11.axis('off')
        ax12.axis('off')
        ax13.axis('off')
        ax11.set_title('img2 Map')
        ax12.set_title('img2 W (N) E')
        ax13.set_title('img2 N (W) S')
        ax4.set_title("img code " + str(classcode) + " connectivity")
        pos01 = ax01.imshow(img1[slice_iz, :, :], cmap='rainbow', vmin=vmin, vmax=vmax)
        ax02.imshow(img1[:, slice_iy, :], cmap='rainbow', vmin=vmin, vmax=vmax)
        ax03.imshow(img1[:, :, slice_ix], cmap='rainbow', vmin=vmin, vmax=vmax)
        fig.colorbar(pos01, cax=axins, label=clblab)
        ax11.imshow(img2[slice_iz, :, :], cmap='rainbow', vmin=vmin, vmax=vmax)
        ax12.imshow(img2[:, slice_iy, :], cmap='rainbow', vmin=vmin, vmax=vmax)
        ax13.imshow(img2[:, :, slice_ix], cmap='rainbow', vmin=vmin, vmax=vmax)
        ax4.plot(lag_xc1, lag_cp1, 'ro-')
        ax4.plot(lag_xc2, lag_cp2, 'b+--')
        ax4.legend(('img1', 'img2'), loc='best')
        ax4.set_xlabel("lag distance [px]", fontsize=14)
        ax4.set_ylabel("Connectivity probability")  # ,fontsize=14
    if ndim == 2:
        fig = plt.figure()
        gs = fig.add_gridspec(3, 5)
        ax00 = fig.add_subplot(gs[0, 2])
        ax01 = fig.add_subplot(gs[0, 0])
        ax11 = fig.add_subplot(gs[0, 1])
        ax4 = fig.add_subplot(gs[1:2, 1:2])
        axins = inset_axes(ax00,
                           width="5%",  # width = 5% of parent_bbox width
                           height="90%",  # height : 50%
                           loc='center left'
                           )
        ax00.axis('off')
        ax01.axis('off')
        ax01.set_title('img1')
        ax11.axis('off')
        ax11.set_title('img2')
        ax4.set_title("img code " + str(classcode) + " connectivity")
        pos01 = ax01.imshow(img1, cmap='rainbow', vmin=vmin, vmax=vmax)
        fig.colorbar(pos01, cax=axins, label=clblab)
        ax11.imshow(img2, cmap='rainbow', vmin=vmin, vmax=vmax)
        ax4.plot(lag_xc1, lag_cp1, 'ro-')
        ax4.plot(lag_xc2, lag_cp2, 'b+--')
        ax4.legend(('img1', 'img2'), loc='best')
        ax4.set_xlabel("lag distance [px]", fontsize=14)
        ax4.set_ylabel("Connectivity probability")  # ,fontsize=14
    fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=0.55, wspace=0.1, hspace=0.5)
    plt.show()
    return


def weighted_lpnorm(array1, array2, p, weights=np.array([]), verb=False):
    if weights.shape != array1.shape:
        weights = np.ones(array1.shape)
    if verb:
        print('weights: ' + np.array2string(weights, precision=2, separator=','))
    ix2keep = np.where((np.isnan(array1) | np.isnan(array2)) == False)
    w = weights[ix2keep] / np.sum(weights[ix2keep])
    L = (np.sum(w * (np.abs(array1[ix2keep] - array2[ix2keep])) ** p)) ** (1 / p)
    return L
