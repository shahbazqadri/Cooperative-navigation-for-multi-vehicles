import scipy.io
import numpy as np
from matplotlib import pyplot as plt

MC_TOTAL = 20
nb_agents = 4
nb_landmarks = 2
T = 50
adj_type = 'd_ring'
#metrics = ['basic', 'Trace_cov', 'Trace_inv_SAM']# 'last_SAM_cov']#, 'min_eig_inv_cov']
metrics = ['Trace_cov', 'Trace_inv_SAM','det_cov','SAM']#, 'basic']#, 'SAM']
colorcode = 'rbgkcy'
linestyle = ['-.', '-',(0, (10, 10)), ':','--']
S = [0,1]#range(nb_agents)
def compute_rmse(err, mc_s=0, mc_e=MC_TOTAL, remove = None):
    # err: MC_TOTAL X nb_agents X x_dim X timesteps

    N = err.shape[0]
    if mc_s == None:
        mc_s = 0
    if mc_e == None:
        mc_e = N
    nb_agents = err.shape[1]
    err_sqr = err ** 2
    if remove == None:
        perr_rmse = np.sqrt(np.sum(err_sqr[mc_s:mc_e,:,0:2,:], axis = (0,1,2)) / N / nb_agents)
        herr_rmse = np.sqrt(np.sum(err_sqr[mc_s:mc_e,:,2:,:], axis = (0,1,2))  / N / nb_agents)
    else:
        perr_sum = 0
        herr_sum = 0
        for i in range(mc_s, mc_e):
            if i not in remove:
                perr_sum = err_sqr[i:i+1,:,0:2,:] + perr_sum
                herr_sum = err_sqr[i:i + 1, :, 2:, :] + herr_sum
        perr_rmse = np.sqrt(np.sum(perr_sum, axis = (0,1,2)) / N / nb_agents)
        herr_rmse = np.sqrt(np.sum(herr_sum, axis = (0,1,2)) / N / nb_agents)
    return perr_rmse, herr_rmse


results = {}
n_sub = 2 + len(metrics)
P = []
COV = []
for s in S:
    for i, metric in enumerate(metrics):
        mc = 4
        filename = str(MC_TOTAL) + '-' + str(nb_agents) + '-' + str(T) + '-' + adj_type + '-' + metric + '-' + str(s) + '-' + str(nb_landmarks) + '-once_w20_LM_sn.mat'
        mat = scipy.io.loadmat(filename)
        est_err = mat['EST'] - mat['TRUTH']
        # P.append(mat['P'])
        # COV.append(mat['COV'])
        est_err[:,:,2:,:] = np.unwrap(est_err[:,:,2:,:])
        if s == 2:
            if metric == 'Trace_cov':
                rmse = compute_rmse(est_err, 0, 20, remove = [4,8,13])
            elif metric == 'Trace_inv_SAM':
                rmse = compute_rmse(est_err, 0, 20, remove = [4,8,13])
            elif metric == 'basic':
                rmse = compute_rmse(est_err, 0, 20, remove = [4,8,13])
            rmse = compute_rmse(est_err)
        elif s == 0:
            rmse = compute_rmse(est_err, 0, 20, remove = [4, 6, 8, 19])
            rmse = compute_rmse(est_err)
        else:
            rmse = compute_rmse(est_err)
        results[metric] = rmse
        #plt.figure(figsize = (9,3))
        plt.figure(s)
        plt.subplot(n_sub,1,1)
        plt.plot(np.arange(0,T,0.1), rmse[0], label =metric, color = colorcode[i], linestyle=linestyle[i])
        plt.subplot(n_sub, 1, 2)
        plt.plot(np.arange(0, T, 0.1), rmse[1], label=metric, color= colorcode[i], linestyle=linestyle[i])
        for ai in range(nb_agents):
            plt.subplot(n_sub, 1, 3+i)
            plt.plot(mat['TRUTH'][mc,ai,0,:], mat['TRUTH'][mc,ai,1,:], color=colorcode[ai], linestyle=linestyle[i])
            plt.grid(True)
            plt.ylim([-500, 500])
            plt.xlim([0, 1600])
            plt.title(metric)
    plt.subplot(n_sub,1,1)
    plt.ylabel("position RMSE")
    plt.grid(True)
    plt.legend()
    plt.subplot(n_sub,1,2)
    plt.ylabel("heading RMSE")
    plt.grid(True)
    plt.legend()
    plt.xlabel("time steps")

plt.show()