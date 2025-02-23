import scipy.io
import numpy as np
from matplotlib import pyplot as plt

MC_TOTAL = 20
nb_agents = 5
nb_landmarks = 0
T = 55
vel = 30
adj_type = 'd_ring'
#metrics = ['basic', 'Trace_cov', 'Trace_inv_SAM']# 'last_SAM_cov']#, 'min_eig_inv_cov']
metrics = ['Trace_inv_SAM', 'Trace_cov']#,'basic']#, 'SAM']
colorcode = 'rbgkcy'
linestyle = ['-.', '-','--', (0, (10, 10)), ':','--']
pos0 = 10 * np.array([[10., 0., 10., 20., 10.], [10., 0., -10., 0., 0.]])
pos0 = pos0[:nb_agents, :nb_agents]
posf = pos0 + np.ones(
                (2, nb_agents)) * vel * 50 * 0.7

S = range(nb_agents)
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
n_sub = 2 #+ len(metrics)
P = []
COV = []
for s in S:
    print(f'leader vehicle {s}:')
    for i, metric in enumerate(metrics):
        mc = 0
        if metric == 'basic':
            T = 50
            filename = str(MC_TOTAL) + '-' + str(nb_agents) + '-' + str(T) + '-' + adj_type + '-' + metric + '-' + str(s) + '-' + str(nb_landmarks) + '-once_w20_LM_newMPC2.mat'
        else:
            T = 50
            if nb_landmarks == 2:
                filename = str(MC_TOTAL) + '-' + str(nb_agents) + '-' + str(T) + '-' + adj_type + '-' + metric + '-' + str(s) + '-' + str(nb_landmarks) + '-once_w20_LM_newMPC1.mat'
            elif nb_landmarks == 0:
                filename = str(MC_TOTAL) + '-' + str(nb_agents) + '-' + str(
                    T) + '-' + adj_type + '-' + metric + '-' + str(s) + '-' + str(
                    nb_landmarks) + '-once_w20_LM_newMPC2.mat'
        mat = scipy.io.loadmat(filename)
        est_err = mat['EST'] - mat['TRUTH']
        # P.append(mat['P'])
        # COV.append(mat['COV'])
        est_err[:,:,2:,:] = np.unwrap(est_err[:,:,2:,:])
        # if s == 2:
        #     if adj_type == 'full' and nb_agents == 3 and nb_landmarks == 2:
        #         if metric == 'Trace_cov':
        #             rmse = compute_rmse(est_err, 0, 20, remove = [4,8,13])
        #         elif metric == 'Trace_inv_SAM':
        #             rmse = compute_rmse(est_err, 0, 20, remove = [4,8,13])
        #         elif metric == 'basic':
        #             rmse = compute_rmse(est_err, 0, 20, remove = [4,8,13])
        #     else:
        #         rmse = compute_rmse(est_err, 0, 20, remove = [6,12])
        # elif s == 0:
        #     if adj_type == 'full' and nb_agents == 3 and nb_landmarks == 2:
        #         rmse = compute_rmse(est_err, 0, 20, remove = [4, 6, 8, 19])
        #     else:
        #         rmse = compute_rmse(est_err, 0, 20, remove = [3,12,14,15])
        # else:
        #     rmse = compute_rmse(est_err, 0, 20, remove=[6])
        rmse = compute_rmse(est_err)
        print(metric, rmse[0][-1])
        results[metric] = rmse
        plt.figure(s, figsize = (10,6))
        plt.rcParams['font.size'] = 12
        plt.rcParams['lines.linewidth'] = 2
        #
        plt.subplot(n_sub,1,1)
        plt.plot(np.arange(0,T,0.1), rmse[0], label =metric, color = colorcode[i], linestyle=linestyle[i])
        plt.ylabel("position RMSE (meters)")
        plt.grid(True)
        plt.title(f'Leader robot: robot {s + 1}')
        plt.legend()
        plt.subplot(n_sub, 1, 2)
        plt.plot(np.arange(0, T, 0.1), rmse[1], label=metric, color= colorcode[i], linestyle=linestyle[i])
        plt.ylabel("heading RMSE (rad)")
        plt.grid(True)
        plt.legend()
        plt.xlabel("time (secs)")
        plt.tight_layout()
        #plt.savefig(str(s) + adj_type +'.png',dpi=600)
        plt.figure(20 + s, figsize=(10, 10))
        LM = mat['LM']
        for l in range(nb_landmarks):
            print('Landmark locations:')
            print(LM[:,l])
        for mc in range(MC_TOTAL):
            plt.rcParams['font.size'] = 20
            plt.rcParams['lines.linewidth'] = 2
            for ai in range(nb_agents):
                plt.subplot(len(metrics), 1, i+1)
                for l in range(nb_landmarks):
                    plt.plot(LM[0,l], LM[1,l], 'rs')
                if ai == s:
                    plt.plot(mat['TRUTH'][mc,ai,0,:], mat['TRUTH'][mc,ai,1,:], color=colorcode[ai], linestyle= '-')#linestyle=linestyle[ai])
                    plt.plot(posf[0,ai], posf[1, ai], color=colorcode[ai+1], marker='D')
                    plt.grid(True)
                    #plt.ylim([-510, 510])
                    #plt.xlim([0, 1000])
                    plt.ylim([-100, 1200])
                    plt.xlim([0, 1300])
                    plt.xlabel('meters')
                    plt.ylabel('meters')
                    plt.annotate(metric, (50, 350),fontsize = 'large')
                    if mc == 0 and i == 0:
                        plt.title(f'Leader robot: robot {s + 1}')
    plt.tight_layout()
    #plt.savefig(str(s) + adj_type + '_traj.png', dpi=600)
plt.show()