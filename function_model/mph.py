import json
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from numpy.random import default_rng
import matplotlib.colors as mcolors
import  cmaps
import uuid
import base64

import common.file_util
from common import file_util
from common.file_util import read_file_to_arr_2d
from common.minio_util import upload_files_minio_bucket_by_dir,upload_files_minio_bucket_by_file


def mph_file(ti_path, path_file):
    ti_arr, ti_x, ti_y = read_file_to_arr_2d(ti_path)
    realization_arr, realization_x, realization_y = read_file_to_arr_2d(path_file)

    # n_levels = max2Dlevels + 1
    n_levels = 3
    seed = 65432
    n_clusters = 10
    nmax_patterns = int(1E4)
    pattern2Dsize = np.asarray([8, 8])
    pattern3Dsize = np.asarray([1, 2, 3])
    patternsize = pattern2Dsize
    dis,file_name_list = dist_kmeans_mph(ti_arr, realization_arr, n_levels, patternsize, n_clusters, nmax_patterns, seed,
                          plot=True, verb=True)
    return dis,file_name_list

def mph_dir(ti_path, path_dir):
    """
    MPH函数
    :param ti_path:
    :param path_dir:
    :return:
    """
    path_dir_name_list = os.listdir(path_dir)
    ti_arr, ti_x, ti_y = read_file_to_arr_2d(ti_path)

    # MPH based dissimilarity parameters
    seed = 65432
    n_clusters = 10
    nmax_patterns = int(1E4)
    pattern2Dsize = np.asarray([8, 8])
    pattern3Dsize = np.asarray([1, 2, 3])

    max2Dlevels = np.min([np.floor(np.log(ti_y / (pattern2Dsize[0] + n_clusters ** (1 / 2))) / np.log(2)),
                          np.floor(np.log(ti_x / (pattern2Dsize[1] + n_clusters ** (1 / 2))) / np.log(2))]).astype(int)

    # 这里把文件写死了 ,换个文件的话需要重新进行统计

    # 基于LOOP_UI里面的方法进行比较
    file_name_dis_map_list = []  # 文件名字和训练图像的映射关系

    # 2023/11/26 文件存储

    for i in range(0, len(path_dir_name_list)):
        temp_path_dir = os.path.join(path_dir, path_dir_name_list[i])
        temp_path_name_list = os.listdir(temp_path_dir)
        for path_index in range(len(temp_path_name_list)):
            temp_path = os.path.join(temp_path_dir, temp_path_name_list[path_index])

            realization_arr, realization_x, realization_y = read_file_to_arr_2d(temp_path)

            # n_levels = max2Dlevels + 1
            n_levels = 3
            patternsize = pattern2Dsize
            if (path_index == 0):
                dis,_ = dist_kmeans_mph(ti_arr, realization_arr, n_levels, patternsize, n_clusters, nmax_patterns, seed,
                                      plot=True, verb=True)
            else:
                dis,_ = dist_kmeans_mph(ti_arr, realization_arr, n_levels, patternsize, n_clusters, nmax_patterns, seed,
                                      plot=False,
                                      verb=False)

            file_name_dis_map_list.append([temp_path, dis])

    file_name_dis_map_arr = np.array(file_name_dis_map_list)
    file_name_dis_map_df = pd.DataFrame(file_name_dis_map_arr, columns=["name", "distance"])
    file_name_dis_map_df.to_csv("file_name_dis_map.csv")


def dist_kmeans_mph(img1, img2, n_levels, patternsize, n_clusters, nmax_patterns, seed, plot=False, verb=False):
    # initialize distance value for incrementation
    d = 0.0
    # get ndim
    ndim = len(img1.shape)
    # 存储文件名字
    file_name_list = []
    for l in range(n_levels + 1):
        rng = np.random.default_rng(2 * seed + l)
        if verb:
            print('Level ' + str(l))
        # get pattern matrix shape
        tmp_shape = list(np.asarray(list(img1.shape)) - patternsize + 1)
        tmp_shape.append(np.prod(patternsize))
        tmp_shape = tuple(tmp_shape)
        # get nb patterns  prod()计算乘积
        npat = np.prod(tmp_shape[:-1])
        # get patterns and sample from img1 and img2
        img1_all_patterns = np.ones(tmp_shape) * np.nan
        img2_all_patterns = np.ones(tmp_shape) * np.nan
        if verb:
            print('Number of possible patterns: ' + str(npat))
        if ndim == 2:
            [ppdim1, ppdim0] = np.meshgrid(np.arange(patternsize[1]), np.arange(patternsize[0]))
            ppdim0 = ppdim0.flatten()
            ppdim1 = ppdim1.flatten()
            for pp in range(np.prod(patternsize)):
                img1_all_patterns[:, :, pp] = img1[ppdim0[pp]:tmp_shape[0] + ppdim0[pp],
                                              ppdim1[pp]:tmp_shape[1] + ppdim1[pp]]
                img2_all_patterns[:, :, pp] = img2[ppdim0[pp]:tmp_shape[0] + ppdim0[pp],
                                              ppdim1[pp]:tmp_shape[1] + ppdim1[pp]]
        elif ndim == 3:
            [ppdim2, ppdim1, ppdim0] = np.meshgrid(np.arange(patternsize[2]), np.arange(patternsize[1]),
                                                   np.arange(patternsize[0]))
            ppdim0 = ppdim0.flatten()
            ppdim1 = ppdim1.flatten()
            ppdim2 = ppdim2.flatten()
            for pp in range(np.prod(patternsize)):
                img1_all_patterns[:, :, :, pp] = img1[ppdim0[pp]:tmp_shape[0] + ppdim0[pp],
                                                 ppdim1[pp]:tmp_shape[1] + ppdim1[pp],
                                                 ppdim2[pp]:tmp_shape[2] + ppdim2[pp]]
                img2_all_patterns[:, :, :, pp] = img2[ppdim0[pp]:tmp_shape[0] + ppdim0[pp],
                                                 ppdim1[pp]:tmp_shape[1] + ppdim1[pp],
                                                 ppdim2[pp]:tmp_shape[2] + ppdim2[pp]]
        img1_all_patterns = np.reshape(img1_all_patterns, (npat, np.prod(patternsize)))
        img2_all_patterns = np.reshape(img2_all_patterns, (npat, np.prod(patternsize)))
        # subsamling the patterns
        if npat > nmax_patterns:
            ix_sub1 = (np.floor(rng.uniform(0, 1, nmax_patterns) * (npat - 1))).astype(int)
            ix_sub2 = (np.floor(rng.uniform(0, 1, nmax_patterns) * (npat - 1))).astype(int)
        else:
            ix_sub1 = np.arange(npat)
            ix_sub2 = np.arange(npat)
        img1_patterns = img1_all_patterns[ix_sub1, :]
        img2_patterns = img2_all_patterns[ix_sub2, :]
        if verb:
            print('Number of sub-sampled patterns: ' + str(len(ix_sub1)))
        del img1_all_patterns, img2_all_patterns, ix_sub1, ix_sub2

        # kmeans clustering of patterns
        kmeans_img1 = KMeans(n_clusters=n_clusters, random_state=0).fit(img1_patterns)
        img1_cluster_id, img1_cluster_size = np.unique(kmeans_img1.labels_, return_counts=True)
        kmeans_img2 = KMeans(n_clusters=n_clusters, random_state=0).fit(img2_patterns)
        img2_cluster_id, img2_cluster_size = np.unique(kmeans_img2.labels_, return_counts=True)
        # find best cluster pairing for mph dist computation
        img_cluster_id_pairs_dist = np.ones((n_clusters, 3)) * np.nan  # cluster_id1, cluster_id2, distance
        cpy_img2_cluster_id = img2_cluster_id + 0
        for c in range(n_clusters):
            tmp_cluster = kmeans_img1.cluster_centers_[img1_cluster_id[c], :]
            tmp_dist = (np.sum((kmeans_img2.cluster_centers_[cpy_img2_cluster_id, :] - tmp_cluster) ** 2,
                               axis=1)) ** 0.5
            img_cluster_id_pairs_dist[c, 0] = img1_cluster_id[c]
            img_cluster_id_pairs_dist[c, 1] = cpy_img2_cluster_id[np.argmin(tmp_dist)]
            img_cluster_id_pairs_dist[c, 2] = tmp_dist[np.argmin(tmp_dist)]
            cpy_img2_cluster_id = np.delete(cpy_img2_cluster_id, np.argmin(tmp_dist))
        # compute distance contribution as density weighted distance between closest best paired clusters
        p1 = img1_cluster_size / np.sum(img1_cluster_size)
        p2 = img2_cluster_size / np.sum(img2_cluster_size)
        weights = (np.abs(p1 - p2) / (p1 + p2))
        # weights_sum = np.sum(img1_cluster_size+img2_cluster_size)
        # weights = img1_cluster_size[(img_cluster_id_pairs_dist[:,0]).astype(int)] + img2_cluster_size[(img_cluster_id_pairs_dist[:,1]).astype(int)]
        # dist_mphc = np.sum( img_cluster_id_pairs_dist[:,2] * weights ) / weights_sum
        dist_mphc = np.sum(((1 + img_cluster_id_pairs_dist[:, 2]) * (1 + weights) - 1))
        d += dist_mphc / n_levels
        if verb:
            print('Distance component: ' + str(dist_mphc / n_levels))
        if plot:
            # plot_kmeans_mph(img1, img2, l, kmeans_img1, kmeans_img2, img_cluster_id_pairs_dist, patternsize,
            #                 uniqueColorScale=False)
            file_name = plot_kmeans_mph(img1, img2, l, kmeans_img1, kmeans_img2, img_cluster_id_pairs_dist, patternsize,
                            uniqueColorScale=True)
            file_name_list.append(file_name)
        if l <= n_levels:
            img1 = stochastic_upscale(img1, seed + l)
            img2 = stochastic_upscale(img2, seed + l)
            del img_cluster_id_pairs_dist, tmp_cluster, tmp_dist, cpy_img2_cluster_id, kmeans_img1, kmeans_img2
            del img1_cluster_id, img1_cluster_size, img2_cluster_id, img2_cluster_size
            del tmp_shape, npat, img1_patterns, img2_patterns
    return d,file_name_list


def plot_kmeans_mph(img1, img2, l, kmeans_img1, kmeans_img2, img_cluster_id_pairs_dist, patternsize,
                    uniqueColorScale=False):
    c10 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[0, 0]).astype(int), :], tuple(patternsize))
    c11 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[1, 0]).astype(int), :], tuple(patternsize))
    c12 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[2, 0]).astype(int), :], tuple(patternsize))
    c13 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[3, 0]).astype(int), :], tuple(patternsize))
    c14 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[4, 0]).astype(int), :], tuple(patternsize))
    c15 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[5, 0]).astype(int), :], tuple(patternsize))
    c16 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[6, 0]).astype(int), :], tuple(patternsize))
    c17 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[7, 0]).astype(int), :], tuple(patternsize))
    c18 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[8, 0]).astype(int), :], tuple(patternsize))
    c19 = np.reshape(kmeans_img1.cluster_centers_[(img_cluster_id_pairs_dist[9, 0]).astype(int), :], tuple(patternsize))
    c20 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[0, 1]).astype(int), :], tuple(patternsize))
    c21 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[1, 1]).astype(int), :], tuple(patternsize))
    c22 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[2, 1]).astype(int), :], tuple(patternsize))
    c23 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[3, 1]).astype(int), :], tuple(patternsize))
    c24 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[4, 1]).astype(int), :], tuple(patternsize))
    c25 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[5, 1]).astype(int), :], tuple(patternsize))
    c26 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[6, 1]).astype(int), :], tuple(patternsize))
    c27 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[7, 1]).astype(int), :], tuple(patternsize))
    c28 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[8, 1]).astype(int), :], tuple(patternsize))
    c29 = np.reshape(kmeans_img2.cluster_centers_[(img_cluster_id_pairs_dist[9, 1]).astype(int), :], tuple(patternsize))
    bc1 = np.bincount(kmeans_img1.labels_)
    bc2 = np.bincount(kmeans_img2.labels_)
    bc2 = bc2[(img_cluster_id_pairs_dist[:, 1]).astype(int)]
    tfs = 8
    fig_m = plt.figure(constrained_layout=True)
    gs = fig_m.add_gridspec(4, 7)
    fm_ax1 = fig_m.add_subplot(gs[:2, :2])
    fm_ax1.set_title('img1 level ' + str(l)), fm_ax1.axis('off')
    fm_ax2 = fig_m.add_subplot(gs[2:, :2])
    fm_ax2.set_title('img2 level ' + str(l)), fm_ax2.axis('off')
    fm_ax10 = fig_m.add_subplot(gs[0, 2])
    fm_ax10.axis('off'), fm_ax10.set_title(str(bc1[0]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[0,0]).astype(int))+' - '+
    fm_ax11 = fig_m.add_subplot(gs[0, 3])
    fm_ax11.axis('off'), fm_ax11.set_title(str(bc1[1]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[1,0]).astype(int))+' - '+
    fm_ax12 = fig_m.add_subplot(gs[0, 4])
    fm_ax12.axis('off'), fm_ax12.set_title(str(bc1[2]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[2,0]).astype(int))+' - '+
    fm_ax13 = fig_m.add_subplot(gs[0, 5])
    fm_ax13.axis('off'), fm_ax13.set_title(str(bc1[3]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[3,0]).astype(int))+' - '+
    fm_ax14 = fig_m.add_subplot(gs[0, 6])
    fm_ax14.axis('off'), fm_ax14.set_title(str(bc1[4]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[4,0]).astype(int))+' - '+
    fm_ax15 = fig_m.add_subplot(gs[1, 2])
    fm_ax15.axis('off'), fm_ax15.set_title(str(bc1[5]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[5,0]).astype(int))+' - '+
    fm_ax16 = fig_m.add_subplot(gs[1, 3])
    fm_ax16.axis('off'), fm_ax16.set_title(str(bc1[6]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[6,0]).astype(int))+' - '+
    fm_ax17 = fig_m.add_subplot(gs[1, 4])
    fm_ax17.axis('off'), fm_ax17.set_title(str(bc1[7]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[7,0]).astype(int))+' - '+
    fm_ax18 = fig_m.add_subplot(gs[1, 5])
    fm_ax18.axis('off'), fm_ax18.set_title(str(bc1[8]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[8,0]).astype(int))+' - '+
    fm_ax19 = fig_m.add_subplot(gs[1, 6])
    fm_ax19.axis('off'), fm_ax19.set_title(str(bc1[9]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[9,0]).astype(int))+' - '+
    fm_ax20 = fig_m.add_subplot(gs[2, 2])
    fm_ax20.axis('off'), fm_ax20.set_title(str(bc2[0]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[0,1]).astype(int))+' - '+
    fm_ax21 = fig_m.add_subplot(gs[2, 3])
    fm_ax21.axis('off'), fm_ax21.set_title(str(bc2[1]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[1,1]).astype(int))+' - '+
    fm_ax22 = fig_m.add_subplot(gs[2, 4])
    fm_ax22.axis('off'), fm_ax22.set_title(str(bc2[2]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[2,1]).astype(int))+' - '+
    fm_ax23 = fig_m.add_subplot(gs[2, 5])
    fm_ax23.axis('off'), fm_ax23.set_title(str(bc2[3]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[3,1]).astype(int))+' - '+
    fm_ax24 = fig_m.add_subplot(gs[2, 6])
    fm_ax24.axis('off'), fm_ax24.set_title(str(bc2[4]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[4,1]).astype(int))+' - '+
    fm_ax25 = fig_m.add_subplot(gs[3, 2])
    fm_ax25.axis('off'), fm_ax25.set_title(str(bc2[5]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[5,1]).astype(int))+' - '+
    fm_ax26 = fig_m.add_subplot(gs[3, 3])
    fm_ax26.axis('off'), fm_ax26.set_title(str(bc2[6]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[6,1]).astype(int))+' - '+
    fm_ax27 = fig_m.add_subplot(gs[3, 4])
    fm_ax27.axis('off'), fm_ax27.set_title(str(bc2[7]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[7,1]).astype(int))+' - '+
    fm_ax28 = fig_m.add_subplot(gs[3, 5])
    fm_ax28.axis('off'), fm_ax28.set_title(str(bc2[8]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[8,1]).astype(int))+' - '+
    fm_ax29 = fig_m.add_subplot(gs[3, 6])
    fm_ax29.axis('off'), fm_ax29.set_title(str(bc2[9]),
                                           fontsize=tfs)  # 'CP '+str((img_cluster_id_pairs_dist[9,1]).astype(int))+' - '+
    camp = 'bwr' #红蓝色
    # camp = cmaps.MPL_RdYlGn
    if uniqueColorScale == True:
        vmin1 = np.amin(img1)
        vmin2 = np.amin(img2)
        vmax1 = np.amax(img1)
        vmax2 = np.amax(img2)
        vmin = np.min([vmin1, vmin2])
        vmax = np.min([vmax1, vmax2])
        fm_ax1.imshow(img1,cmap=camp,vmin=vmin, vmax=vmax)
        fm_ax2.imshow(img2,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax10.imshow(c10,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax11.imshow(c11,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax12.imshow(c12,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax13.imshow(c13,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax14.imshow(c14,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax15.imshow(c15,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax16.imshow(c16,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax17.imshow(c17,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax18.imshow(c18,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax19.imshow(c19,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax20.imshow(c20,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax21.imshow(c21,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax22.imshow(c22,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax23.imshow(c23,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax24.imshow(c24,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax25.imshow(c25,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax26.imshow(c26,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax27.imshow(c27,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax28.imshow(c28,cmap=camp, vmin=vmin, vmax=vmax)
        fm_ax29.imshow(c29,cmap=camp, vmin=vmin, vmax=vmax)
    else:
        fm_ax1.imshow(img1,cmap=camp)  # ,vmin=vmin,vmax=vmax
        fm_ax2.imshow(img2,cmap=camp)  # ,vmin=vmin,vmax=vmax
        fm_ax10.imshow(c10,cmap=camp)
        fm_ax11.imshow(c11,cmap=camp)
        fm_ax12.imshow(c12,cmap=camp)
        fm_ax13.imshow(c13,cmap=camp)
        fm_ax14.imshow(c14,cmap=camp)
        fm_ax15.imshow(c15,cmap=camp)
        fm_ax16.imshow(c16,cmap=camp)
        fm_ax17.imshow(c17,cmap=camp)
        fm_ax18.imshow(c18,cmap=camp)
        fm_ax19.imshow(c19,cmap=camp)
        fm_ax20.imshow(c20,cmap=camp)
        fm_ax21.imshow(c21,cmap=camp)
        fm_ax22.imshow(c22,cmap=camp)
        fm_ax23.imshow(c23,cmap=camp)
        fm_ax24.imshow(c24,cmap=camp)
        fm_ax25.imshow(c25,cmap=camp)
        fm_ax26.imshow(c26,cmap=camp)
        fm_ax27.imshow(c27,cmap=camp)
        fm_ax28.imshow(c28,cmap=camp)
        fm_ax29.imshow(c29,cmap=camp)
    # plt.show()
    # 生成唯一的URL
    uuid1 = file_util.get_uuid()
    # file_name = str(uuid1) + "level"+str(l)+".png"
    file_name ="level"+str(l)+".png"
    file_url = file_util.file_dir_name + file_name  # 文件先上传
    # 将文件 上传到minio中

    plt.savefig(file_url, dpi=1200)
    url = upload_files_minio_bucket_by_file(file_url,"level"+str(l)+".png" ,file_util.user_name)
    plt.show()
    file_stream = file_util.return_img_stream(file_url)
    print("======",l)
    return url


def stochastic_upscale(mx, seed):
    rng = default_rng(seed)
    ndim = len(mx.shape)
    ux_shape = tuple(np.floor(np.asarray(mx.shape) / 2).astype(int))
    reductionfactor = 2 ** ndim
    tmp_shape = list(np.floor(np.asarray(list(mx.shape)) / 2).astype(int))
    tmp_shape.append(reductionfactor)
    tmp_shape = tuple(tmp_shape)
    tmp = np.ones(tmp_shape) * np.nan
    v = np.array([0, 1])
    if ndim == 2:
        ny, nx = mx.shape
        [dx, dy] = np.meshgrid(v, v)
        dx = dx.flatten().astype(int)
        dy = dy.flatten().astype(int)
    elif ndim == 3:
        nz, ny, nx = mx.shape
        [dx, dy, dz] = np.meshgrid(v, v, v)
        dx = dx.flatten().astype(int)
        dy = dy.flatten().astype(int)
        dz = dz.flatten().astype(int)
    else:
        return -1
    for i in range(reductionfactor):
        if ndim == 2:
            tmp2 = mx[dy[i]:ny + dy[i]:2, dx[i]:nx + dx[i]:2]
            tmp[:, :, i] = tmp2[0:ux_shape[0], 0:ux_shape[1]]
            del tmp2
        elif ndim == 3:
            tmp2 = mx[dz[i]:nz + dz[i]:2, dy[i]:ny + dy[i]:2, dx[i]:nx + dx[i]:2]
            tmp[:, :, :, i] = tmp2[0:ux_shape[0], 0:ux_shape[1], 0:ux_shape[2]]
            del tmp2
    ix2 = np.reshape(np.floor(rng.uniform(0, reductionfactor - 1e-12, np.prod(ux_shape))).astype(int),
                     ux_shape).flatten()
    ix1 = np.arange(np.prod(ux_shape)).flatten()
    tmp = np.reshape(tmp, (np.prod(ux_shape), reductionfactor))
    upscaled_mx = np.reshape(tmp[ix1, ix2], ux_shape)
    return upscaled_mx
