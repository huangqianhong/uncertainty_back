import numpy as np
from matplotlib import pyplot as plt
from numpy.random import default_rng
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from common.file_util import read_file_to_arr_2d

base = np.e

def variogram_file(ti_path, realization_path):
    img1, ti_x, ti_y = read_file_to_arr_2d(ti_path)
    img2, refer_x, refer_y = read_file_to_arr_2d(realization_path)
    n_levels = 3
    patternsize = [9, 9]
    n_clusters = 10
    nmax_patterns = 10000
    seed = 65432
    d = -1

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
    d, lag_distance_list, lagxc_list = dist_experimental_variogram_categorical_add_line_data(img1, img2, xxx, yyy, zz,
                                                                                             nblags, maxh, maxnbsamples,
                                                                                             pnorm,
                                                                                             seed,
                                                                                             label="", verb=False,
                                                                                             plot=True, cmap='viridis',
                                                                                             slice_ix=0,
                                                                                             slice_iy=0, slice_iz=0)

    return d, lag_distance_list, lagxc_list



def dist_experimental_variogram_categorical_add_line_data(img1, img2, xxx, yyy, zzz, nblags, maxh, maxnbsamples, pnorm,
                                                          seed,
                                                          label="", verb=False, plot=False, cmap='viridis', slice_ix=0,
                                                          slice_iy=0,
                                                          slice_iz=0):
    classvalues = np.unique(np.hstack((img1, img2)))
    nbclasses = len(classvalues)
    d = 0
    lag_distance_list = []
    lagxc_list = []  # 包含两个地方的边界线

    dbyclass = np.zeros(nbclasses)
    for c in range(nbclasses):
        current_class = classvalues[c]
        tmp1 = ((img1 == current_class) * 1).astype(int)
        tmp2 = ((img2 == current_class) * 1).astype(int)
        curr_label = label + " " + str(current_class)
        if verb:
            print('img1 ' + curr_label)
        [lag_xc1, lag_sv1, lag_ct1] = experimental_variogram(tmp1, xxx, yyy, zzz, nblags, maxh, maxnbsamples, seed,
                                                             verb=verb)
        if verb:
            print('img2 ' + curr_label)
        [lag_xc2, lag_sv2, lag_ct2] = experimental_variogram(tmp2, xxx, yyy, zzz, nblags, maxh, maxnbsamples, seed,
                                                             verb=verb)
        w = 2 / (lag_xc1 + lag_xc2)
        dbyclass[c] = weighted_lpnorm(lag_sv1, lag_sv2, pnorm, weights=w, verb=verb)
        d += dbyclass[c] ** pnorm
        if verb:
            print('distance ' + curr_label + ": " + str(dbyclass))
        if plot == True:
            plot_experimental_variograms(img1, img2, lag_xc1, lag_xc2, lag_sv1, lag_sv2, curr_label, cmap, slice_ix,
                                         slice_iy, slice_iz)
            plt.plot(lag_xc1, lag_sv1, color="red", linestyle="solid", linewidth=1.5, marker="*", mec='r', mfc='w',
                     markersize=12,
                     label="店铺销售趋势")
            plt.plot(lag_xc2, lag_sv2, color="red", linestyle="solid", linewidth=1.5, marker="*", mec='r', mfc='w',
                     markersize=12,
                     label="店铺销售趋势")
            plt.show(block=False)
        lag_xc1 = lag_xc1.astype(int)
        lag_distance_list = lag_xc1.tolist()
        lagsv_list = [lag_sv1.tolist(), lag_sv2.tolist()]

    d = d ** (1 / pnorm)

    return d, lag_distance_list, lagsv_list

def dist_experimental_variogram(img1,img2,xxx,yyy,zzz,nblags,maxh,maxnbsamples,pnorm,seed,categ=False,label="",verb=False,plot=False,cmap='viridis',slice_ix=0,slice_iy=0,slice_iz=0):
    d=-1
    if categ==False:
        d = dist_experimental_variogram_continuous(img1,img2,xxx,yyy,zzz,nblags,maxh,maxnbsamples,pnorm,seed,label=label,verb=verb,plot=plot,cmap=cmap,slice_ix=slice_ix,slice_iy=slice_iy,slice_iz=slice_iz)
    elif categ==True:
        d = dist_experimental_variogram_categorical(img1,img2,xxx,yyy,zzz,nblags,maxh,maxnbsamples,pnorm,seed,label=label,verb=verb,plot=plot,cmap=cmap,slice_ix=slice_ix,slice_iy=slice_iy,slice_iz=slice_iz)
    return d


def plot_experimental_variograms(img1,img2,lag_xc1,lag_xc2,lag_sv1,lag_sv2,label,cmap,slice_ix,slice_iy,slice_iz):
    ndim = len(img1.shape)
    vmin = np.min([np.nanmin(img1),np.nanmin(img2)])
    vmax = np.max([np.nanmax(img1),np.nanmax(img2)])
    if ndim==3:
        fig = plt.figure()
        gs = fig.add_gridspec(2,7)
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
        ax4.set_title("experimental variogram")
        pos01=ax01.imshow(img1[slice_iz,:,:],cmap=cmap,vmin=vmin,vmax=vmax)
        ax02.imshow(img1[:,slice_iy,:],cmap=cmap,vmin=vmin,vmax=vmax)
        ax03.imshow(img1[:,:,slice_ix],cmap=cmap,vmin=vmin,vmax=vmax)
        fig.colorbar(pos01,cax=axins,label=label)
        ax11.imshow(img2[slice_iz,:,:],cmap=cmap,vmin=vmin,vmax=vmax)
        ax12.imshow(img2[:,slice_iy,:],cmap=cmap,vmin=vmin,vmax=vmax)
        ax13.imshow(img2[:,:,slice_ix],cmap=cmap,vmin=vmin,vmax=vmax)
        ax4.plot(lag_xc1, lag_sv1, 'ro-')
        ax4.plot(lag_xc2, lag_sv2, 'b+--')
        ax4.legend(('img1', 'img2'),loc='best')
        ax4.set_xlabel("$h$ - lag distance [px]",fontsize=14)
        ax4.set_ylabel("$\gamma(h)$") #,fontsize=14
    if ndim==2:
        fig = plt.figure()
        gs = fig.add_gridspec(1,5)
        ax00 = fig.add_subplot(gs[0, 2])
        ax01 = fig.add_subplot(gs[0, 0])
        ax11 = fig.add_subplot(gs[0,1])
        ax4 = fig.add_subplot(gs[0, 3:])
        axins = inset_axes(ax00,
                           width="10%",  # width = 5% of parent_bbox width
                           height="90%",  # height : 50%
                           loc='center left'
                           )
        ax00.axis('off')
        ax01.axis('off')
        ax01.set_title('img1')
        ax11.axis('off')
        ax11.set_title('img2')
        ax4.set_title("experimental variogram")
        pos01=ax01.imshow(img1,cmap=cmap,vmin=vmin,vmax=vmax)
        fig.colorbar(pos01,cax=axins,label=label)
        ax11.imshow(img2,cmap=cmap,vmin=vmin,vmax=vmax)
        ax4.plot(lag_xc1, lag_sv1, 'ro-')
        ax4.plot(lag_xc2, lag_sv2, 'b+--')
        ax4.legend(('img1', 'img2'),loc='best')
        ax4.set_xlabel("$h$ - lag distance [px]",fontsize=14)
        ax4.set_ylabel("$\gamma(h)$") #,fontsize=14
    fig.subplots_adjust(left=0.0, bottom=0.0, right=2.0, top=0.55, wspace=0.1, hspace=0.5)
    plt.show()
    return


def dist_experimental_variogram_continuous(img1,img2,xxx,yyy,zzz,nblags,maxh,maxnbsamples,pnorm,seed,label="",verb=False,plot=False,cmap='viridis',slice_ix=0,slice_iy=0,slice_iz=0):
    d=0
    if verb:
        print('img1')
    [lag_xc1, lag_sv1, lag_ct1] = experimental_variogram(img1,xxx,yyy,zzz,nblags,maxh,maxnbsamples,seed,verb=verb)
    if verb:
        print('img2')
    [lag_xc2, lag_sv2, lag_ct2] = experimental_variogram(img2,xxx,yyy,zzz,nblags,maxh,maxnbsamples,seed,verb=verb)
    if verb:
        print('distance computation')
    w = 2/(lag_xc1+lag_xc2)
    d = weighted_lpnorm(lag_sv1,lag_sv2,pnorm,weights=w,verb=verb)
    if plot==True:
        plot_experimental_variograms(img1,img2,lag_xc1,lag_xc2,lag_sv1,lag_sv2,label,cmap,slice_ix,slice_iy,slice_iz)
    return d


def experimental_variogram(array, xxx, yyy, zzz, nblags, maxh, maxnbsamples, seed, verb=False):
    rng = default_rng(seed)
    ndim = len(array.shape)
    mask = (np.isnan(array) == False)
    nbvaliddata = np.sum((mask.flatten() == True) * 1)
    if ndim == 3:
        [nz, ny, nx] = array.shape
    elif ndim == 2:
        [ny, nx] = array.shape
        nz = 1
    if verb:
        print(str(ndim) + 'D data - experimental semi-variogram computation')
    laglim = np.linspace(0, maxh, nblags + 1)
    lag_xc = np.ones(nblags) * np.nan
    lag_sv = np.ones(nblags) * np.nan
    lag_ct = np.ones(nblags) * np.nan
    if nbvaliddata <= maxnbsamples:
        sv_samples_ix = np.arange(nbvaliddata)
    else:
        sv_samples_ix = (np.round(rng.uniform(0, 1, maxnbsamples) * (nbvaliddata - 1))).astype(int)

    sv_samples_val = np.reshape(array[mask], (nbvaliddata, 1))[sv_samples_ix]
    sv_samples_xxx = np.reshape(xxx[mask], (nbvaliddata, 1))[sv_samples_ix]
    sv_samples_yyy = np.reshape(yyy[mask], (nbvaliddata, 1))[sv_samples_ix]
    sv_samples_zzz = np.reshape(zzz[mask], (nbvaliddata, 1))[sv_samples_ix]

    # compute distance and square diff between sampled pair of points
    sv_dist = np.ones(np.round(len(sv_samples_ix) * (len(sv_samples_ix) - 1) / 2).astype(int)) * np.nan
    sv_sqdf = np.ones(np.round(len(sv_samples_ix) * (len(sv_samples_ix) - 1) / 2).astype(int)) * np.nan
    k = 0
    for i in range(len(sv_samples_ix)):
        for j in np.arange(i):
            sv_dist[k] = ((sv_samples_xxx[i] - sv_samples_xxx[j]) ** 2 + (
                        sv_samples_yyy[i] - sv_samples_yyy[j]) ** 2 + (
                                      sv_samples_zzz[i] - sv_samples_zzz[j]) ** 2) ** 0.5
            sv_sqdf[k] = (sv_samples_val[i] - sv_samples_val[j]) ** 2
            k += 1
    # for each lag
    for l in range(nblags):
        # identify sampled pairs belonging to the lag
        lag_lb = laglim[l]
        lag_ub = laglim[l + 1]
        ix = np.where((sv_dist >= lag_lb) & (sv_dist < lag_ub))
        # count, experimental semi vario value and center of lag cloud
        lag_ct[l] = len(ix[0])
        lag_xc[l] = np.mean(sv_dist[ix])
        lag_sv[l] = np.mean(sv_sqdf[ix]) * 0.5
    return lag_xc, lag_sv, lag_ct


def dist_experimental_variogram_categorical(img1,img2,xxx,yyy,zzz,nblags,maxh,maxnbsamples,pnorm,seed,label="",verb=False,plot=False,cmap='viridis',slice_ix=0,slice_iy=0,slice_iz=0):
    classvalues = np.unique(np.hstack((img1,img2)))
    nbclasses = len(classvalues)
    d=0
    dbyclass = np.zeros(nbclasses)
    for c in range(nbclasses):
        current_class = classvalues[c]
        tmp1 = ((img1==current_class)*1).astype(int)
        tmp2 = ((img2==current_class)*1).astype(int)
        curr_label = label + " " + str(current_class)
        if verb:
            print('img1 '+curr_label)
        [lag_xc1, lag_sv1, lag_ct1] = experimental_variogram(tmp1,xxx,yyy,zzz,nblags,maxh,maxnbsamples,seed,verb=verb)
        if verb:
            print('img2 '+curr_label)
        [lag_xc2, lag_sv2, lag_ct2] = experimental_variogram(tmp2,xxx,yyy,zzz,nblags,maxh,maxnbsamples,seed,verb=verb)
        w = 2/(lag_xc1+lag_xc2)
        dbyclass[c] = weighted_lpnorm(lag_sv1,lag_sv2,pnorm,weights=w,verb=verb)
        d+=dbyclass[c]**pnorm
        if verb:
            print('distance '+ curr_label +": "+str(dbyclass))
        if plot==True:
            plot_experimental_variograms(img1,img2,lag_xc1,lag_xc2,lag_sv1,lag_sv2,curr_label,cmap,slice_ix,slice_iy,slice_iz)
    d = d**(1/pnorm)
    return d



def mxdist_experimental_variogram_continuous(img_all,xxx,yyy,zzz,nblags,maxh,maxnbsamples,pnorm,seed,verb=False):
    nb_models = img_all.shape[-1]
    ndim = len(img_all.shape)-1
    mxd = np.zeros((nb_models,nb_models))
    lag_xc_all = np.ones((nblags,nb_models))*np.nan
    lag_sv_all = np.ones((nblags,nb_models))*np.nan
    for i in range(nb_models):
        if ndim==3:
            img = img_all[:,:,:,i]
        elif ndim==2:
            img = img_all[:,:,i]
        if verb:
            print('experimental vario model '+str(i))
        [lag_xc_all[:,i], lag_sv_all[:,i], _ ] = experimental_variogram(img,xxx,yyy,zzz,nblags,maxh,maxnbsamples,seed,verb=verb)
    if verb:
        print('distance computation')
    for i in range(nb_models):
        for j in range(i):
            if verb:
                print('i = '+str(i)+' - j = '+str(j))
            w = 2/(lag_xc_all[:,i]+lag_xc_all[:,j])
            d = weighted_lpnorm(lag_sv_all[:,i],lag_sv_all[:,j],pnorm,weights=w,verb=verb)
            mxd[i,j] = d
            mxd[j,i] = d
    return mxd

def mxdist_experimental_variogram_categorical(img_all,xxx,yyy,zzz,nblags,maxh,maxnbsamples,pnorm,seed,categval,verb=False):
    # categval = np.unique(img_all)
    nbclasses = len(categval)
    nb_models = img_all.shape[-1]
    ndim = len(img_all.shape)-1
    mxd = np.zeros((nb_models,nb_models))
    mxdbyclass = np.zeros((nb_models,nb_models,nbclasses))
    lag_xc_all = np.ones((nblags,nb_models,nbclasses))*np.nan
    lag_sv_all = np.ones((nblags,nb_models,nbclasses))*np.nan
    for c in range(nbclasses):
        current_class = categval[c]
        curr_label = " class " + str(current_class)
        for i in range(nb_models):
            if ndim==3:
                tmp = img_all[:,:,:,i]
            elif ndim==2:
                tmp = img_all[:,:,i]
            img = ((tmp==current_class)*1).astype(int)
            if verb:
                print('experimental vario model '+str(i)+' - '+curr_label)
            [lag_xc_all[:,i,c], lag_sv_all[:,i,c], _ ] = experimental_variogram(img,xxx,yyy,zzz,nblags,maxh,maxnbsamples,seed,verb=verb)

        if verb:
            print('distance computation')
        for i in range(nb_models):
            for j in range(i):
                if verb:
                    print('i = '+str(i)+' - j = '+str(j))
                w = 2/(lag_xc_all[:,i,c]+lag_xc_all[:,j,c])
                d = weighted_lpnorm(lag_sv_all[:,i,c],lag_sv_all[:,j,c],pnorm,weights=w,verb=verb)
                mxdbyclass[i,j,c] = d
                mxdbyclass[j,i,c] = d
    mxd = (np.sum(mxdbyclass**pnorm,axis=2))**(1/pnorm)
    return mxd


def weighted_lpnorm(array1,array2,p,weights=np.array([]),verb=False):
    if weights.shape!=array1.shape:
        weights=np.ones(array1.shape)
    if verb:
        print('weights: '+np.array2string(weights, precision=2, separator=','))
    ix2keep=np.where((np.isnan(array1) | np.isnan(array2))==False)
    w=weights[ix2keep]/np.sum(weights[ix2keep])
    L=(np.sum(w*(np.abs(array1[ix2keep]-array2[ix2keep]))**p))**(1/p)
    return L
