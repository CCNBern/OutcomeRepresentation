import numpy as np

from ProbabilityFunctions import getProbabilityEstimates_group, prepareProbabilitiesToPlot_mostLikely_lessLikely, \
    groupSubjects_perCondition, loadProbabilities, saveProbabilities, concatenateAllConditions_probabilities

import copy

from scipy.fft import rfft, rfftfreq
from mne.time_frequency import psd_array_multitaper, psd_array_welch

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

seed = 42

def loadNPfile(filename):
    print('filename to load: ' + filename + '.npy')
    data = np.load(filename + '.npy', allow_pickle=True)
    print('shape data: ', data.shape)
    return data

def loadDict_fromNPY(filename):
    print('filename to load: ' + filename + '.npy')
    mydict = np.load(filename + '.npy', allow_pickle=True)

    print('probs_all keys: ', str(mydict[()].keys()))
    return mydict[()]

def frequencyAnalysis(probabilities, sub_s_list, conditions, fs, isPermuted, keyList, psd_func=None, fmin=1,fmax=20):

    data_ft_group, data_ft_singleTrial_group = {}, {}
    ft_frequencies = None
    probabilities_cond = {}


    if len(conditions) == 1:
        for sub_s in probabilities.keys():
            probabilities_cond[sub_s] = probabilities[sub_s][conditions[0]]#['mostLikely']

    else:
        probabilities_cond = concatenateAllConditions_probabilities(probabilities, conditions, isPermuted, keyList) # double check for less likely!!!!!!!

    del probabilities


    for s_ind, sub_s in enumerate(sub_s_list):
        print('sub_s: ', sub_s)
        data_ft_group[sub_s], data_ft_singleTrial_group[sub_s] = {}, {}

        for key in probabilities_cond[sub_s].keys(): # likeliness
            data_allOut = probabilities_cond[sub_s][key]

            if isPermuted == False:

                # loop over mostLikelyOutcomes (could be 1 or 2) to compute ft
                for outInd in range(data_allOut.shape[1]):

                    if psd_func != None:
                        # prepare the parameters of the functions
                        kwargs = dict(fmin=fmin, fmax=fmax, n_jobs=-1, verbose=False, sfreq=fs)

                        if psd_func ==  psd_array_multitaper:
                            kwargs['normalization'] = 'full'
                            kwargs['adaptive'] = False#True


                        elif psd_func == psd_array_welch:
                            kwargs['average'] =  'median' #mean
                            hamming_window = int(fs / 2)
                            overlap = int(hamming_window / 2)
                            kwargs['n_overlap'] = overlap
                            n_fft = int(hamming_window)
                            kwargs['n_fft'] = n_fft

                        # compute fouier transformation
                        psd, ft_frequencies = psd_func(data_allOut[:,outInd,:], **kwargs) # #  psd_array_multitaper
                        #print('ft_frequencies: ', ft_frequencies)
                        #psd = 10 * np.log10(psd)  # convert to dB
                        psd = 10 * np.log10(psd, out=psd, where=psd > 0)  # convert to dB #  / np.max(psd)

                    else:
                        psd = np.abs(rfft(data_allOut[:, outInd, :]))
                        ft_frequencies = rfftfreq(data_allOut.shape[-1], 1 / fs)

                    # initialize array to store ft of all outcomes
                    if outInd == 0:
                        data_ft_allOut = np.zeros((data_allOut.shape[0], data_allOut.shape[1], ft_frequencies.shape[0]))  # (ntrials x nOutcomes x ft_frequnecies)

                    data_ft_allOut[:, outInd, :] = psd

            else:

                # loop over permutations
                for permInd in range(data_allOut.shape[0]):
                    # loop over mostLikelyOutcomes (could be 1 or 2) to compute ft
                    for outInd in range(data_allOut.shape[2]):
                        if psd_func != None:
                            # prepare the parameters of the functions
                            kwargs = dict(fmin=fmin, fmax=fmax, n_jobs=1, verbose=0, sfreq=fs)

                            if psd_func == psd_array_multitaper:
                                kwargs['normalization'] = 'full'
                                kwargs['adaptive'] = True

                            elif psd_func == psd_array_welch:
                                kwargs['average'] = 'median'  # mean
                                hamming_window = int(fs / 2)
                                overlap = int(hamming_window / 2)
                                kwargs['n_overlap'] = overlap
                                n_fft = int(hamming_window)
                                kwargs['n_fft'] = n_fft

                            # compute fouier transformation
                            psd, ft_frequencies = psd_func(data_allOut[permInd, :, outInd, :], **kwargs)  # #  psd_array_multitaper
                            psd = 10 * np.log10(psd, out=psd, where=psd > 0)  # convert to dB #/ np.max(psd)

                        else:
                            psd = np.abs(rfft(data_allOut[permInd, :, outInd, :]))
                            ft_frequencies = rfftfreq(data_allOut.shape[-1], 1 / fs)

                        # initialize array to store ft of all permutations & outcomes
                        if permInd == 0:
                            data_ft_allOut = np.zeros((data_allOut.shape[0], data_allOut.shape[1], data_allOut.shape[2],
                                                       ft_frequencies.shape[0]))  # (nPermutations x ntrials x nOutcomes x ft_frequnecies)

                        data_ft_allOut[permInd, :, outInd, :] = psd
                        del psd

                # store single trials
                data_allOut_singleTrial = copy.deepcopy(data_ft_allOut)
                # take mean over the permutations
                data_ft_allOut = np.mean(data_ft_allOut, axis=0)

            if isPermuted == False: # for permuted it's already stored see above
                # store single trials
                data_allOut_singleTrial = copy.deepcopy(data_ft_allOut)

            # take mean over the outcomes
            data_ft = np.mean(data_ft_allOut, axis=1)  # -> per condition in per subject
            del data_ft_allOut
            # take mean over trials and store it in group list

            if len(keyList) > 1: #for the analysis of most Likely, lessLikely
                data_ft_group[sub_s][key] = np.mean(data_ft, axis=0)
                data_ft_singleTrial_group[sub_s][key] = data_allOut_singleTrial

            else: # for the analysis of mostLikely only
                data_ft_group[sub_s] = np.mean(data_ft, axis=0)
                data_ft_singleTrial_group[sub_s] = data_allOut_singleTrial

            del data_ft


    return data_ft_group, ft_frequencies, data_ft_singleTrial_group




def computeAreaUnderCurve(freqs, data, sigPoints):
    auc = np.trapz(data[sigPoints], freqs[sigPoints])
    return auc



def calculateDiffMetric(metricType,data_ft_group,data_ft_group_perm, freqs, isAbs, sigPoints):

    metric_real, metric_perm, metric_delta = {}, {}, {}


    if metricType == 'Auc':
        for sub_s in data_ft_group.keys():

            # auc real
            metric_real[sub_s] = computeAreaUnderCurve(freqs, data_ft_group[sub_s], sigPoints)

            # auc perm
            metric_perm[sub_s] = computeAreaUnderCurve(freqs, data_ft_group_perm[sub_s], sigPoints)

            # compute delta auc
            if isAbs == True:
                metric_delta[sub_s] = abs(metric_perm[sub_s] - metric_real[sub_s])  # abs
            else:
                metric_delta[sub_s] = (metric_perm[sub_s] - metric_real[sub_s])

        #  convert it to array
        metric_delta_arr = np.array(list(metric_delta.values()))

        return metric_delta_arr
    else:
         return None




def linkingWithBehaviour(metricType, data_ft_group, data_ft_group_perm, freqs, isAbs, cond, base_path_beh,
                         split_num, resultsPath, usePartialData, sigPoints):

    # read accuracy from file
    metric_delta_arr = calculateDiffMetric(metricType, data_ft_group, data_ft_group_perm, freqs, isAbs, sigPoints)


    fontSize_labels, fontSize_ticks, fontSize_title = 8, 10, 25
    multiplier = 1.35


    if usePartialData == True:
        if cond == 'all':
            beh_acc_group = np.genfromtxt(base_path_beh + 'acc_group_twoHalves_mostLikely.csv',
                                          delimiter=',')  # 1st, 2nd
            print("beh_acc_group: ", beh_acc_group)

            if split_num == 0:
                data_toPlot = {metricType: metric_delta_arr, 'Accuracy': beh_acc_group[:, 0]}
            elif split_num == 1:
                data_toPlot = {metricType: metric_delta_arr, 'Accuracy': beh_acc_group[:, 1]}


        else:
            beh_acc_group = np.genfromtxt(base_path_beh + 'acc_group_twoHalves_cfVScd_mostLikely.csv',
                                          delimiter=',')  # cf_1st, cd_1st, cf_2nd, cd_2nd
            print("beh_acc_group: ", beh_acc_group)

            if split_num == 0:
                if 'cd' in cond:
                    data_toPlot = {metricType + '_cd': metric_delta_arr, 'Accuracy_cd': beh_acc_group[:, 1]}
                elif 'cf' in cond:
                    data_toPlot = {metricType + '_cf': metric_delta_arr, 'Accuracy_cf': beh_acc_group[:, 0]}
            elif split_num == 1:
                if 'cd' in cond:
                    data_toPlot = {metricType + '_cd': metric_delta_arr, 'Accuracy_cd': beh_acc_group[:, -1]}
                elif 'cf' in cond:
                    data_toPlot = {metricType + '_cf': metric_delta_arr, 'Accuracy_cf': beh_acc_group[:, -2]}

            else:
                print('Error occured while accessing the split!!')
                data_toPlot = None
    else:

        beh_acc_group = np.genfromtxt(base_path_beh + 'acc_group_mostLikely.csv', delimiter=',')  # cf, cd
        print("beh_acc_group: ", beh_acc_group)

        if 'cd' in cond:
            data_toPlot = {metricType + '_cd': metric_delta_arr, 'Accuracy_cd': beh_acc_group[:, 1]}
        elif 'cf' in cond:
            data_toPlot = {metricType + '_cf': metric_delta_arr, 'Accuracy_cf': beh_acc_group[:, 0]}


        else:
            print('Error occured!!')
            data_toPlot = None


    df_toPlot = pd.DataFrame(data_toPlot)

    if cond == 'all':
        g = sns.lmplot(data=df_toPlot, x="Accuracy", y=metricType, seed=seed)
    else:
        if 'cd' in cond:
            y_label = metricType + "_cd"
            x_label =  "Accuracy_cd"
            color = '#C40E94'
        elif 'cf' in cond:
            y_label = metricType + "_cf"
            x_label = "Accuracy_cf"
            color = '#008A09'

        g = sns.lmplot(data=df_toPlot, x=x_label, y=y_label,
                       line_kws={'color': color, 'linewidth':2*multiplier}, scatter_kws={"color": color}, seed=seed, ci=None)

        g.set(xlabel=None, ylabel=None)



    plt.xlim(xmin=0.50, xmax=1)
    plt.ylim(ymin=-3, ymax=24)


    ax = plt.gca()
    fig_final = plt.gcf()
    fig_final.set_size_inches(3.54, 3.54 * 0.75)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5 * multiplier)
    ax.spines['bottom'].set_color('black')
    ax.spines["left"].set_linewidth(1.5 * multiplier)
    ax.spines['left'].set_color('black')
    plt.xticks(fontsize=fontSize_ticks * multiplier)
    plt.yticks(fontsize=fontSize_ticks * multiplier)
    ax.yaxis.tick_left()
    ax.yaxis.set_tick_params(width=1.5 * multiplier, length=3 * multiplier, colors='black')
    ax.xaxis.tick_bottom()
    ax.xaxis.set_tick_params(width=1.5 * multiplier, length=3 * multiplier, colors='black')

    for tick in ax.get_xticklabels():
        tick.set_fontname("Helvetica Neue")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Helvetica Neue")

    xticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
    plt.xticks(xticks)

    yticks = [0, 8, 16, 24]
    plt.yticks(yticks)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.5*multiplier)
    ax.spines['bottom'].set_color('black')
    ax.spines["left"].set_linewidth(1.5*multiplier)
    ax.spines['left'].set_color('black')
    plt.xticks(fontsize=fontSize_ticks*multiplier)
    plt.yticks(fontsize=fontSize_ticks*multiplier)
    ax.yaxis.tick_left()
    ax.yaxis.set_tick_params(width=1.5*multiplier, length=3*multiplier, colors='black')
    ax.xaxis.tick_bottom()
    ax.xaxis.set_tick_params(width=1.5*multiplier, length=3*multiplier, colors='black')

    for tick in ax.get_xticklabels():
        tick.set_fontname("Helvetica Neue")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Helvetica Neue")


    plt.tight_layout()
    plt.rcParams['savefig.dpi'] = 600

    plt.xlabel('Accuracy', fontweight='bold')
    plt.ylabel(metricType, fontweight='bold')


    if isAbs == True:

        print(resultsPath + 'Abs_' + metricType + '_accuracy_' + cond + '_' + metricType + '_mostLikely_freq_sigPts.png')

        if usePartialData == True:
            plt.savefig(resultsPath + 'Abs_' + metricType + '_accuracy_' + cond + '_' + str(
                split_num) + '_' + metricType + '_mostLikely_freq_sigPts.png')

        else:
            plt.savefig(resultsPath + 'Abs_' + metricType + '_accuracy_' + cond + '_' + metricType + '_mostLikely_freq_sigPts.png')
    else:
        if usePartialData == True:
            plt.savefig(resultsPath + metricType + '_accuracy_' + cond + '_' + str(
                split_num) + '_' + metricType + '_mostLikely_freq_sigPts.png')
        else:
            plt.savefig(resultsPath + metricType + '_accuracy_' + cond + '_' + metricType + '_mostLikely_freq_sigPts.png')

