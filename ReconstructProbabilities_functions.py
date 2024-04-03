import pickle
import numpy as np
from PlotFunctions import plotProbabilityEstimates
from testingFunctions import testingIsAllZero


def reconstructProbabilities(sub_s, data, times, clf_filename, result_filename, pred_prob_filename, pred_prob_Plotfilename,
                  labels, noutcomes=3, isPermuted=False):

    print('----->  Subject: ', sub_s)
    #data = epochs.get_data()
    print(data.shape)

    # load the model from disk
    clf = pickle.load(open(clf_filename, 'rb'))

    # predict & estimate the probabilities
    result = np.zeros((data.shape[0], len(times)))
    pred_prob = np.zeros((data.shape[0], noutcomes, len(times)))
    for i in range(len(times)):
        result[:, i] = clf.predict(data[:, :, i])
        pred_prob[:, :, i] = clf.predict_proba(data[:, :, i])

        # check if the predictions and probabilities are not all zero
        testingIsAllZero(result[:, i])
        testingIsAllZero(pred_prob[:, :, i])

    if isPermuted == False:  # save results and probs if not permuted; permuted will be saved later with all permutations
        np.save(result_filename, result)
        np.save(pred_prob_filename, pred_prob)
        print('pred_prob: ', pred_prob)
        print('pred_prob[0,:,0]: \n', pred_prob[0, :, 0])
        print('result: ', result)
        plotProbabilityEstimates(pred_prob, times, pred_prob_Plotfilename, sub_s, labels)

    return result, pred_prob


def reconstructProbabilities_chanceLevel(sub_s, epoch_data, times, clf_filename_base, pred_prob_filename, labels, noutcomes=3, maxPermutations=None):

    # initialize pred_prob_perm array # initialize pred_prob_perm array
    #epoch_data = epochs.get_data() # need shape of this for initialization in the next line
    pred_prob_perm = np.zeros((maxPermutations, epoch_data.shape[0], noutcomes, epoch_data.shape[2]))

    if maxPermutations != None:
        for nper in range(maxPermutations):
            clf_filename = clf_filename_base + str(nper) + '.sav'

            res, pred_prob = reconstructProbabilities(sub_s, epoch_data, times, clf_filename, None, pred_prob_filename,
                                           None, labels, noutcomes=3, isPermuted=True)


            pred_prob_perm[nper,:,:,:] = pred_prob


        # save all maxPermutations for probabilities
        print('pred_prob_perm SHAPE: ', pred_prob_perm.shape)
        np.save(pred_prob_filename, pred_prob_perm)


    else:
        print('No permutation!!!')

    return pred_prob_perm

