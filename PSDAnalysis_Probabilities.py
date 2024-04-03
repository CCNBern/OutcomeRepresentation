import numpy as np

from DataFunctions import getTimes, find_nearestTimePoint, generateFilename
from PlotFunctions import plotFrequnecySpectrum
from testingFunctions import testingType
from ProbabilityFunctions import getProbabilityEstimates_group
from StatFunctions import compareTwoConditions_wilcoxon
from mne.time_frequency import psd_array_multitaper
from PSDAnalysis_functions import loadNPfile, frequencyAnalysis

import os


seed = 42


if __name__ == '__main__':

    # Initialize constants
    basepath = ''
    eegpath = basepath + ''
    resultsPath = basepath + ''
    base_path_beh = ''
    fs=256
    dataset_nameExt = ''

    sub_s_list = []


    time_indices = "44_77"

    obj_names = ['cf_obj', 'cd_obj1', 'cd_obj2']
    context_names = ['', '_season1', '_season2'] # obj_alone, season1, season2
    contexts_list = [None, 11, 12] # None: obj_only
    times = getTimes(sub_s_list[-1], eegpath, fs=fs)

    # For analysis of beginning and end separately
    usePartialData = False

    if usePartialData == False:
        splits = [None]
        splitSize = None
    else:
        splits = [0,1] #None#1
        splitSize = 1/2 #None#1/3

    isPermuted = True#True#True
    correction = True

    psd_func = psd_array_multitaper

    conditions_toCompare = ['all-cf']

    #  ------------------------ GROUP RESULTS ---------------------------------------------------------------------


    prob_estimates_group_final_allSplits = []
    for split_num in splits:

        for condition_toTest in conditions_toCompare:

            if condition_toTest == 'all-cf':

                conditions_all = ['cf_obj', 'cf_obj_season1', 'cf_obj_season2']

            elif condition_toTest == 'all-cf_obj':
                conditions_all = ['cf_obj']

            elif condition_toTest == 'all-cf_objCon':
                conditions_all = ['cf_obj_season1', 'cf_obj_season2']

            elif condition_toTest == 'all-cd_obj':

                conditions_all = ['cd_obj1', 'cd_obj2']

            elif condition_toTest == 'all-cd_objCon':

                conditions_all = ['cd_obj1_season1', 'cd_obj1_season2', 'cd_obj2_season1', 'cd_obj2_season2']

            elif condition_toTest == 'all-cd':

                conditions_all = ['cd_obj1', 'cd_obj2', 'cd_obj1_season1', 'cd_obj1_season2', 'cd_obj2_season1',
                                  'cd_obj2_season2']


            elif condition_toTest == 'all':
                conditions_all = ['cf_obj', 'cf_obj_season1', 'cf_obj_season2',
                                  'cd_obj1', 'cd_obj1_season1', 'cd_obj1_season2',
                                  'cd_obj2', 'cd_obj2_season1', 'cd_obj2_season2']


            pred_prob_filename_allSubs = generateFilename(resultsPath, isGroup=True, taskName='Probabilities',
                                                          nSubs=len(sub_s_list), subjectId=None,
                                                          time_indices=time_indices,
                                                          conditions=None,
                                                          isComparisonConditions=False,
                                                          filenamePrefix='predProbabilities_preplay_GROUP_ALL',
                                                          isPermuted=False, isGroupAvg=False,
                                                          dataset_nameExt=dataset_nameExt,
                                                          usePartialData=usePartialData,
                                                          splitSize=splitSize, split_num=split_num,
                                                          isComparisonBetweenSplits=False)

            psd_real_filename = pred_prob_filename_allSubs + '_' + condition_toTest + '-PSD' #[:-4]
            freq_filename = pred_prob_filename_allSubs + '_' + condition_toTest + '-freq'


            # check if psd already computed

            if os.path.isfile(psd_real_filename + '.npy') == True and os.path.isfile(freq_filename + '.npy') == True:
                print(psd_real_filename + ' exits! Loading...')
                data_ft_group_arr  = loadNPfile(psd_real_filename)

                print(freq_filename + ' exits! Loading...')
                ft_frequencies = loadNPfile(freq_filename)

            else:

                prob_estimates_group = getProbabilityEstimates_group(pred_prob_filename_allSubs, sub_s_list,
                                                                     obj_names, context_names, contexts_list,
                                                                     time_indices, resultsPath, base_path_beh, times,
                                                                     isPermuted=False, type='probs', usePartialData=usePartialData,
                                                                     split_num=split_num, splitSize=splitSize,
                                                                     dataset_nameExt=dataset_nameExt)


                testingType(prob_estimates_group, dict)



                # Compute power spectral density
                data_ft_group, ft_frequencies, data_ft_singleTrial_group = frequencyAnalysis(prob_estimates_group, sub_s_list,
                                                                  conditions_all, fs,
                                                                  isPermuted=False, psd_func=psd_func, keyList=['mostLikely'])

                np.save(psd_real_filename + '_dict' + ".npy", data_ft_group, allow_pickle=True)
                np.save(psd_real_filename + '_singleTrial_dict' + ".npy", data_ft_singleTrial_group, allow_pickle=True)

                # convert dict of subjects with psd data of given condition
                data_ft_group_arr = np.array(list(data_ft_group.values()))
                assert np.any(data_ft_group_arr != np.nan), 'Nan value'
                print('data_ft_group_arr shape: ', data_ft_group_arr.shape)

                del data_ft_group
                # store current split's results in a list to later plot
                # save

                np.save(psd_real_filename, data_ft_group_arr, allow_pickle=True)
                np.save(freq_filename, ft_frequencies, allow_pickle=True)

            # Plot group mean

            plotFilename_freq = generateFilename(resultsPath, isGroup=True, taskName='Probabilities',
                                                 nSubs=len(sub_s_list), subjectId=None,
                                                 time_indices=time_indices,
                                                 conditions=condition_toTest,
                                                 isComparisonConditions=False,
                                                 filenamePrefix='predProbabilities_preplay_FREQ',
                                                 isPermuted=False, isGroupAvg=True,
                                                 dataset_nameExt=dataset_nameExt,
                                                 usePartialData=usePartialData,
                                                 splitSize=splitSize, split_num=split_num,
                                                 isComparisonBetweenSplits=False, isPlotFilename=True)






            if isPermuted == True:

                pred_prob_filename_allSubs_perm = generateFilename(resultsPath, isGroup=True, taskName='Probabilities',
                                                                   nSubs=len(sub_s_list), subjectId=None,
                                                                   time_indices=time_indices,
                                                                   conditions=None,
                                                                   isComparisonConditions=False,
                                                                   filenamePrefix='predProbabilities_preplay_GROUP_ALL',
                                                                   isPermuted=True, isGroupAvg=False,
                                                                   dataset_nameExt=dataset_nameExt,
                                                                   usePartialData=usePartialData,
                                                                   splitSize=splitSize, split_num=split_num,
                                                                   isComparisonBetweenSplits=False)


                psd_perm_filename = pred_prob_filename_allSubs_perm + '_' + condition_toTest + '-PSD'
                freq_filename_perm = pred_prob_filename_allSubs_perm + '_' + condition_toTest + '-freq'


                if os.path.isfile(psd_perm_filename+'.npy') == True and os.path.isfile(freq_filename + ".npy") == True:
                    print(psd_perm_filename +'.npy' + ' exits! Loading...')
                    data_ft_group_perm_arr = loadNPfile(psd_perm_filename)

                    print(freq_filename + ' exits! Loading...')
                    ft_frequencies_perm = loadNPfile(freq_filename)

                else:

                    prob_estimates_group_perm = getProbabilityEstimates_group(pred_prob_filename_allSubs_perm, sub_s_list,
                                                                              obj_names, context_names,
                                                                              contexts_list, time_indices, resultsPath,
                                                                              base_path_beh, times, isPermuted=True, type='probs',
                                                                              usePartialData=usePartialData,
                                                                              split_num=split_num, splitSize=splitSize,
                                                                              dataset_nameExt=dataset_nameExt)




                    testingType(prob_estimates_group_perm, dict)

                    # Compute power spectral density
                    data_ft_group_perm, ft_frequencies_perm, data_ft_singleTrial_group_perm = frequencyAnalysis(prob_estimates_group_perm,
                                                                                sub_s_list,
                                                                                conditions_all, fs, isPermuted=True,
                                                                                psd_func=psd_func, keyList=['mostLikely'])

                    del prob_estimates_group_perm

                    np.save(psd_perm_filename + '_dict', data_ft_group_perm, allow_pickle=True)
                    np.save(psd_perm_filename + '_singleTrial_dict' + ".npy", data_ft_singleTrial_group_perm,
                            allow_pickle=True)

                    # convert dict of subjects with psd data of given condition
                    data_ft_group_perm_arr = np.array(list(data_ft_group_perm.values()))
                    del data_ft_group_perm

                    assert np.any(data_ft_group_perm_arr != np.nan), 'Nan value'

                    # save
                    np.save(psd_perm_filename, data_ft_group_perm_arr, allow_pickle=True)
                    np.save(freq_filename_perm, ft_frequencies_perm, allow_pickle=True)

                # Plot group mean
                plotFilename_freq_perm = generateFilename(resultsPath, isGroup=True, taskName='Probabilities',
                                                     nSubs=len(sub_s_list), subjectId=None,
                                                     time_indices=time_indices,
                                                     conditions=condition_toTest,
                                                     isComparisonConditions=False,
                                                     filenamePrefix='predProbabilities_preplay_FREQ',
                                                     isPermuted=True, isGroupAvg=True,
                                                     dataset_nameExt=dataset_nameExt,
                                                     usePartialData=usePartialData,
                                                     splitSize=splitSize, split_num=split_num,
                                                     isComparisonBetweenSplits=False, isPlotFilename=True)


                # real vs permuted

                plotFilename_freq_realVSperm = generateFilename(resultsPath, isGroup=True, taskName='Probabilities',
                                                          nSubs=len(sub_s_list), subjectId=None,
                                                          time_indices=time_indices,
                                                          conditions=condition_toTest,
                                                          isComparisonConditions=False,
                                                          filenamePrefix='predProbabilities_preplay_FREQ_realVSperm',
                                                          isPermuted=True, isGroupAvg=True,
                                                          dataset_nameExt=dataset_nameExt,
                                                          usePartialData=usePartialData,
                                                          splitSize=splitSize, split_num=split_num,
                                                          isComparisonBetweenSplits=False, isPlotFilename=True)


                p_vals_all, significantPoints = compareTwoConditions_wilcoxon(data_ft_group_arr, data_ft_group_perm_arr,
                                                                              startTimeInd_forCorrecting=0,
                                                                              isAutocorrelation=False,
                                                                              endTimeInd_forCorrecting=None, correcting=correction)

                print('significant frequencies: ', ft_frequencies[significantPoints])

                plotFrequnecySpectrum([data_ft_group_arr, data_ft_group_perm_arr], ft_frequencies_perm,
                                      plotFilename_freq_realVSperm, labels=[condition_toTest + '- Real', condition_toTest + '- Permuted'],
                                      condition=conditions_toCompare, sub_s_list=sub_s_list,
                                      significantPoints=significantPoints, usePartialData=usePartialData,
                                      split_num=split_num, groupMean=False, resultsPath=resultsPath, time_indices=time_indices)






                del data_ft_group_arr, data_ft_group_perm_arr


