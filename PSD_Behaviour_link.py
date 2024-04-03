from DataFunctions import loadNPfile
from DataFunctions import getTimes, find_nearestTimePoint, generateFilename
from PSD_Behaviour_Link_Functions import calculateDiffMetric, linkingWithBehaviour
from PSDAnalysis_functions import loadDict_fromNPY
from StatFunctions import compareTwoConditions_wilcoxon
import numpy as np

seed = 42


# -------------------- END OF FUNCTIONS -------------------


if __name__ == '__main__':

    # Initialize constants
    basepath = ''
    eegpath = basepath + ''
    resultsPath = basepath + ''
    base_path_beh = ''
    fs = 256
    dataset_nameExt = ''

    sub_s_list = []  #

    time_indices = "44_77"  # , "44_110"] # "78_110"#NEW: early time window: 44_77, late window: 78_110, big window: 44_110 #

    obj_names = ['cf_obj', 'cd_obj1', 'cd_obj2']
    context_names = ['', '_season1', '_season2']  # obj_alone, season1, season2
    contexts_list = [None, 11, 12]  # None: obj_only

    times = getTimes(sub_s_list[-1], eegpath, fs=fs)

    usePartialData = False # True for analysis of beginning and end separately

    metricType = 'Auc'
    isAbs = True

    if usePartialData == False:
        splits = [None]
        splitSize = None
    else:
        splits = [0, 1]
        splitSize = 1 / 2

    isPermuted = True

    correcting = True # False for no correction
    forMatrixVis = False
    condition_toTest  = 'all-cd_objCon' #'all-cd_objCon'# 'all-cf'

    #  ------------------------ GROUP RESULTS ---------------------------------------------------------------------

    prob_estimates_group_final_allSplits = []

    for split_num in splits:

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

        psd_real_filename = pred_prob_filename_allSubs + '_' + condition_toTest + '-PSD_dict'
        freq_filename = pred_prob_filename_allSubs + '_' + condition_toTest + '-freq'

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

        psd_perm_filename = pred_prob_filename_allSubs_perm + '_' + condition_toTest + '-PSD_dict'




        data_ft_group = loadDict_fromNPY(psd_real_filename)
        data_ft_group_arr = np.array(list(data_ft_group.values()))
        data_ft_group_perm_tmp = loadDict_fromNPY(psd_perm_filename)
        data_ft_group_perm = {}

        # if dict has dics values -> for likeliness
        if type(data_ft_group_perm_tmp[sub_s_list[0]]) == dict:

            for sub in data_ft_group_perm_tmp.keys():# loop over
                data_ft_group_perm[sub] = data_ft_group_perm_tmp[sub]['mostLikely']
        else:
            data_ft_group_perm = data_ft_group_perm_tmp

        data_ft_group_perm_arr  = np.array(list(data_ft_group_perm.values()))
        ft_frequencies = loadNPfile(freq_filename)



        #if condition_toTest == 'all-cf' or condition_toTest == 'all-cd': # use non corrected p vals to decide the period of calculation for correlation

        p_vals_all, significantPoints = compareTwoConditions_wilcoxon(data_ft_group_arr, data_ft_group_perm_arr,
                                                                      startTimeInd_forCorrecting=0,
                                                                      isAutocorrelation=False,
                                                                      endTimeInd_forCorrecting=None,
                                                                      correcting=correcting)


        linkingWithBehaviour(metricType, data_ft_group, data_ft_group_perm, ft_frequencies, isAbs, condition_toTest,
                             base_path_beh,split_num, resultsPath, usePartialData, significantPoints, isLessLikely=False,
                             correcting=correcting, sub_s_list=sub_s_list)

        if forMatrixVis == True:
            # for summary matrix
            metric_delta_arr = calculateDiffMetric(metricType, data_ft_group, data_ft_group_perm, ft_frequencies, isAbs, significantPoints)

            # print('metric_delta_arr: ', metric_delta_arr)
            # print('avg metric_delta_arr: ', np.mean(metric_delta_arr))
            filename = resultsPath + 'PSD_' + condition_toTest
            if correcting == True:
                if usePartialData == True:
                    filename += '_split=' + str(split_num)
                filename += 'avgDelta-corretedPvals.txt'
            else:
                if usePartialData == True:
                    filename += '_split=' + str(split_num)
                filename += 'avgDelta-uncorretedPvals.txt'

            print('metric_delta_arr final: ', metric_delta_arr)
            print('mean metric_delta_arr: ', np.mean(metric_delta_arr))
            with open(filename, 'w') as f:
                f.write(str(np.mean(metric_delta_arr)))



        del data_ft_group, data_ft_group_perm


        if forMatrixVis: # use corrected p vals to decide the period of calculation to visualize as matrix

            p_vals_all, significantPoints = compareTwoConditions_wilcoxon(data_ft_group_arr, data_ft_group_perm_arr,
                                                                          startTimeInd_forCorrecting=0,
                                                                          isAutocorrelation=False,
                                                                          endTimeInd_forCorrecting=None,
                                                                          correcting=correcting)

            metric_delta_arr = calculateDiffMetric(metricType, data_ft_group[()], data_ft_group_perm[()],
                                                   ft_frequencies, isAbs, significantPoints)

            print('Average delta: ', np.mean(metric_delta_arr))

            with open(resultsPath + 'psd_' + condition_toTest + '_split=' + str(split_num) + 'avgDelta.txt', 'w') as f:
                f.write(str(np.mean(metric_delta_arr)))

            del data_ft_group, data_ft_group_perm

