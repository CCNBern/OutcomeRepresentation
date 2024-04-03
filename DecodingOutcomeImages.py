#!/usr/bin/env python
# coding: utf-8

# ## Decoding outcomes
import os
import pylab, numpy as np, seaborn as sns, matplotlib.pyplot as plt
from scipy.stats import ttest_rel, sem
from DataFunctions import loadEpochs
from PlotFunctions import plotScores_subject
from DecodingFunctions import decodeImages, decodingOutcomeImages_withLocalizer

seed = 42

#decodingOption: 0, 1, 2, 3, 4
# 0: decode images in localizer task,
# 1: decode outcomes with classifiers trained on localizer (time-point by time-point),
# 2: decode outcomes using the peak classification timepoint in localizer decoding
# 3: decode outcomes with classifiers trained on outcomes in main task,
# 4: decode outcomes with classifiers trained on localizer data with a training window,

def main(sub_s_list, eegpath, resultsPath, decodingOption, trainingTimeIndex=None, trainingWindowIndices=None, fs=None):

    print('sub_s_list: ', sub_s_list)
    print('-----------******* Decoding type: ', decodingOption)
    for sub_s in sub_s_list:
        print('----->  Subject: ', sub_s)
        # create path for classification results of current subject
        if not os.path.isdir(resultsPath + sub_s + '/'):
            # Create the directory
            os.mkdir(resultsPath + sub_s + '/')
            os.mkdir(resultsPath + sub_s + '/Classifiers')

            print("Directory for classifiers created!!")


        path = eegpath + sub_s + '/'
        eeg_filename = path + str(sub_s) + "_preproc-epoch_nodetrend_nobc_avgRef_AR.fif"

        epochs = loadEpochs(eeg_filename, fs) # loading resampled data


        # Sanity check: plot erp of each condition and compare them with the plots of data later

        if decodingOption != 3: # uses localzier data
            # Localizer data
            epochs_loc = epochs['CF_out_loc', 'CD_out1_loc', 'CD_out2_loc']

            data_loc = epochs_loc.get_data()  # EEG signals: n_epochs, n_meg_channels, n_times
            labels_loc = epochs_loc.events[:, -1]  # 'CF_out_loc', 'CD_out1_loc', 'CD_out2_loc'
            labels_loc = ((labels_loc % 1000) / 10).astype('int')  # 40, 41, 42

            # check labels
            print('Unique labels: ', np.unique(labels_loc))
            print('Shape of all data: ' + str(data_loc.shape) + ' AND labels: ' + str(labels_loc.shape))


        if decodingOption != 0: # uses main task data
            # Outcomes
            epochs_outcome = epochs['CF_out_corr', 'CF_out_incorr', 'CD_out1_corr', 'CD_out1_incorr', 'CD_out2_corr', 'CD_out2_incorr']
            data_outcome = epochs_outcome.get_data()  # EEG signals: n_epochs, n_meg_channels, n_times
            labels_outcome = epochs_outcome.events[:, -1]
            labels_outcome = (labels_outcome / 100).astype('int')  # 40, 41, 42
            # check labels
            print('Unique labels outcome: ', np.unique(labels_outcome))
            print('Shape of all data outcome: ' + str(data_outcome.shape) + ' AND labels: ' + str(labels_outcome.shape))

            print('-------------------------------------------------------')
            print('-------------------------------------------------------')

        #plot filenames
        if decodingOption == 0:
            scores = decodeImages(data_loc, labels_loc)
            scoresFilename = resultsPath + sub_s + '/' + str(sub_s) + "_MVPA_pointByPoint_loc"
            filename1 = resultsPath + sub_s + '/' + str(sub_s) + '_scores_loc.png'
            #filename2 = resultsPath + sub_s + '/' + str(sub_s) + '_scores_sm=' + str(sm_n) + '_loc.png'


        elif decodingOption == 1:

            scores = decodingOutcomeImages_withLocalizer(data_loc, labels_loc, data_outcome, labels_outcome, epochs.times)
            scoresFilename = resultsPath + sub_s + '/' + str(sub_s) + "_MVPA_pointByPoint_trainLoc_testOutcome"
            filename1 = resultsPath + sub_s + '/' + str(sub_s) + '_scores_trainLoc_testOutcome.png'
            #filename2 = resultsPath + sub_s + '/' + str(sub_s) + '_scores_sm=' + str(sm_n) + '_trainLoc_testOutcome.png'


        elif decodingOption == 2:

            if trainingTimeIndex == None:
                print('Training time point is not defined!\n Cannot do decoding!!!!')

            else:
                print('Training Time Point: ', epochs.times[trainingTimeIndex])
                # save the model to disk
                clf_filename = resultsPath + sub_s + '/Classifiers/clf_ind='+str(trainingTimeIndex)+'.sav'
                scores = decodingOutcomeImages_withLocalizer(data_loc, labels_loc, data_outcome, labels_outcome, epochs.times, trainingTimeIndex, None, clf_filename)
                scoresFilename = resultsPath + sub_s + '/' + str(sub_s) + '_MVPA_pointByPoint_trainLocPeak_testOutcome_ind='+str(trainingTimeIndex)
                filename1 = resultsPath + sub_s + '/' + str(sub_s) + '_scores_trainLocPeak_testOutcome_ind='+str(trainingTimeIndex)+'.png'
                #filename2 = resultsPath + sub_s + '/' + str(sub_s) + '_scores_sm=' + str(sm_n) + '_trainLocPeak_testOutcome_ind='+str(trainingTimeIndex)+'.png'


        elif decodingOption == 3:

            scores = decodeImages(data_outcome, labels_outcome)

            scoresFilename = resultsPath + sub_s + '/' + str(sub_s) + "_MVPA_pointByPoint_trainOutcome_testOutcome"
            filename1 = resultsPath + sub_s + '/' + str(sub_s) + '_scores_trainOutcome_testOutcome.png'
            #filename2 = resultsPath + sub_s + '/' + str(sub_s) + '_scores_sm=' + str(sm_n) + '_trainOutcome_testOutcome.png'

        elif decodingOption == 4:

            if trainingWindowIndices == None:
                print('Training time point is not defined!\n Cannot do decoding!!!!')

            else:
                print('trainingWindowIndices: ', trainingWindowIndices)
                print('Training Time window: ', epochs.times[trainingWindowIndices])
                clf_filename = resultsPath + sub_s + '/Classifiers/clf_ind=' + str(trainingWindowIndices[0]) + '_' + str(trainingWindowIndices[1]) + '.sav'
                scores = decodingOutcomeImages_withLocalizer(data_loc, labels_loc, data_outcome, labels_outcome, epochs.times, None, trainingWindowIndices, clf_filename)
                scoresFilename = resultsPath + sub_s + '/' + str(sub_s) + '_MVPA_pointByPoint_trainLocPeak_testOutcome_ind='+str(trainingWindowIndices[0])+'_'+str(trainingWindowIndices[1])
                filename1 = resultsPath + sub_s + '/' + str(sub_s) + '_scores_trainLocPeak_testOutcome_ind='+str(trainingWindowIndices[0]) +'_' +str(trainingWindowIndices[1])+'.png'
                #filename2 = resultsPath + sub_s + '/' + str(sub_s) + '_scores_sm=' + str(sm_n) + '_trainLocPeak_testOutcome_ind='+str(trainingWindowIndices[0])+'_'+str(trainingWindowIndices[1])+'.png'


        print('+++++++++++++++++++   scores shape: ', scores.shape)
        # save scores
        # scores_group.append(scores)
        np.save(scoresFilename, scores)
        # Plot without smoothing
        plotScores_subject(scores, epochs.times, filename1, sub_s, decodingOption)

        print('scores_out: ', scores)
        print('scores_out avg: ', np.mean(scores, axis=0))


if __name__ == '__main__':

    fs= 256
    basepath = ''
    eegpath = basepath + ''
    resultsPath = basepath + ''

    sub_s_list = []


    # decode localizer images
    scores_loc = main(sub_s_list, eegpath, resultsPath, decodingOption=0, trainingTimeIndex=None, fs=fs)
    trainingTimeIndex = np.argmax(np.mean(scores_loc, axis=0))
    print('max scores_loc avg index: ', trainingTimeIndex)

    
    # decode outcomes with localizer task
    scores_out_withLoc = main(sub_s_list, eegpath, resultsPath, decodingOption=1, trainingTimeIndex=None, fs=fs)
    main(sub_s_list, eegpath, resultsPath, decodingOption=1, trainingTimeIndex=None, fs=fs)
    
    # decode outcomes with localizer at the max performance point
    scores_out_withLoc_peakPoint = main(sub_s_list, eegpath, resultsPath, decodingOption=2, trainingTimeIndex=trainingTimeIndex, fs=fs)
    
    
    # decode outcomes with main task
    scores_out = main(sub_s_list, eegpath, resultsPath, decodingOption=3, trainingTimeIndex=None, fs=fs)
    main(sub_s_list, eegpath, resultsPath, decodingOption=3, trainingTimeIndex=None, fs=fs)

    # decode localizer images using a training window
    training_windows = [[44,77]]

    for trainingWindowIndices in training_windows:
        print('trainingWindowIndices: ', trainingWindowIndices)

        # decode outcomes with localizer using  training window
        main(sub_s_list, eegpath, resultsPath, decodingOption=4, trainingTimeIndex=None, trainingWindowIndices=trainingWindowIndices, fs=fs)

