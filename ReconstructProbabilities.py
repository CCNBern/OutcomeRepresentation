import numpy as np

from DataFunctions import loadEpochs, getContextDependence, getProbabilities_experiment, getLabels, splitData_beginningVSend
from ReconstructProbabilities_functions import decodePreplay
from testingFunctions import testingIsEmpty


# Set parameters
seed = 42
fs= 256

basepath = ''
eegpath = basepath + ''
resultsPath = basepath + ''
training_windows = ["44_77"]

sub_s_list = []

usePartialData = True # True for analyzing the beginning or end  of the experiment
if usePartialData:
    splitSize = 1/2 # split data by halves

dataset_nameExt = ''


for timeInd in training_windows:
    print('timeInd: ', timeInd)
    for sub_s in sub_s_list:
        print('------------ Probability reconstruction: ' + sub_s + ' ---------------')
        base_path_beh  = '' + sub_s + '/'


        probabilities_file = base_path_beh + 'probabilities.csv'
        probabilities_exp = getProbabilities_experiment(probabilities_file)

        object_ContextFree, outcome_ContextFree, outcome_ContextFree_eventcodes, objects_ContextDep, outcomes_ContextDep, outcomes_ContextDep_eventcodes = getContextDependence(probabilities_exp)

        labels = getLabels(outcome_ContextFree_eventcodes)



        ### Load data ###

        path = eegpath + sub_s + '/'
        eeg_filename = path + str(sub_s) + "" + dataset_nameExt+".fif"

        epochs = loadEpochs(eeg_filename, fs)

        # Get the conditions of interest
        conditions_all = epochs.event_id.keys() # get all events in the epochs
        testingIsEmpty(conditions_all)

        conditions_allMain = [c for c in conditions_all if 'loc' not in c]


        # choose the conditions of interest for preplay analysis --> obj alone or obj+season
        conditions_preplay = [c for c in conditions_all if 'obj' in c]
        testingIsEmpty(conditions_preplay)


        # object alone and object & season
        for cond in conditions_preplay:
            clf_filename = resultsPath + 'SingleSubjectResults/' + sub_s + '/Classifiers/clf_ind=' + str(timeInd) + '.sav'


            # compute the preplay
            if usePartialData == False: # use all data

                # prepare data for preplay analysis
                data = epochs[cond].get_data()

                if len(dataset_nameExt) > 0: # correct or incorrect
                    result_filename = resultsPath + 'SingleSubjectResults/' + sub_s + '/' + dataset_nameExt + '/'

                    pred_prob_filename = resultsPath + 'SingleSubjectResults/' + sub_s + '/' + dataset_nameExt + '/'

                    pred_prob_Plotfilename = resultsPath + 'SingleSubjectResults/' + sub_s + '/' + dataset_nameExt + '/'

                else:
                    result_filename = resultsPath + 'SingleSubjectResults/' + sub_s + '/all/'

                    pred_prob_filename = resultsPath + 'SingleSubjectResults/' + sub_s + '/all/'

                    pred_prob_Plotfilename = resultsPath + 'SingleSubjectResults/' + sub_s + '/all/'


                result_filename += str(sub_s) + 'predictions_preplay_' + cond.lower() + '_' + timeInd + dataset_nameExt

                pred_prob_filename += str(sub_s) + 'predProbabilities_preplay_' + cond.lower() + '_' + timeInd + \
                                      dataset_nameExt

                pred_prob_Plotfilename += str(sub_s) + 'predProbabilities_preplay_' + cond.lower() + '_' + timeInd + \
                                          dataset_nameExt + '.png'


                if len(data) > 0:
                    decodePreplay(sub_s, data, epochs.times, clf_filename, result_filename, pred_prob_filename, pred_prob_Plotfilename, labels, noutcomes=3)
                else:
                    np.save(result_filename, None)
                    np.save(pred_prob_filename, None)
                    print('Data size is not sufficient!!')
            else:

                # prepare data for preplay analysis

                data_toTest = splitData_beginningVSend(epochs, cond, sub_s, conditions_allMain) # splitSize

                for sp in range(len(data_toTest)):
                    result_filename = resultsPath + 'SingleSubjectResults/' + sub_s + '/Probabilities/' + timeInd + \
                                      '/splitted/splitSize=' + '%.2f' % splitSize + '/' + str(sub_s) + \
                                      'predictions_preplay_' + cond.lower() + '_' + timeInd + '_split_' + str(sp) + \
                                      '_size=' + str(splitSize) + dataset_nameExt

                    pred_prob_filename = resultsPath + 'SingleSubjectResults/' + sub_s + '/Probabilities/' + timeInd + \
                                         '/splitted/splitSize=' + '%.2f' % splitSize  + '/' + str(sub_s) + \
                                         'predProbabilities_preplay_' + cond.lower() + '_' + timeInd + '_split_' + \
                                         str(sp) + '_size=' + str(splitSize) + dataset_nameExt

                    pred_prob_Plotfilename = resultsPath + 'SingleSubjectResults/' + sub_s + '/Probabilities/' + timeInd + \
                                             '/splitted/splitSize=' + '%.2f' % splitSize + '/' + str(sub_s) + \
                                             'predProbabilities_preplay_' + cond.lower() + '_' + timeInd + '_split_' +\
                                             str(sp) + '_size=' + str(splitSize) + dataset_nameExt + '.png'

                    decodePreplay(sub_s, data_toTest[sp], epochs.times, clf_filename, result_filename,
                                  pred_prob_filename, pred_prob_Plotfilename, labels, noutcomes=3)


