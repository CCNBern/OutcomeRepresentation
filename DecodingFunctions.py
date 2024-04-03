from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mne.decoding import SlidingEstimator, cross_val_multiscore, LinearModel
from sklearn.metrics import confusion_matrix

import pickle
import numpy as np

from testingFunctions import testingIsAllZero, testingIsAllSame

seed = 42

def decodeImages(data2class, labels):

    # train classifiers
    clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(solver='liblinear')))

    time_decod = SlidingEstimator(clf, n_jobs=1, scoring='accuracy', verbose=True)
    scores_all_data = cross_val_multiscore(time_decod, data2class, labels, cv=5, n_jobs=1)

    # check if all scores are zero
    testingIsAllZero(scores_all_data)

    print('Parameters of the classsifier:')
    print('C = ', clf.get_params('linearmodel__model__C'))
    print('Penalty = ', clf.get_params('linearmodel__model__penalty'))

    return scores_all_data


# train classifiers on localizer task and test if they generalize to outcome images

def decodingOutcomeImages_withLocalizer(data_loc, labels_loc, data_outcome, labels_outcome, times,
                                        trainingTimeIndex=None, trainingWindow=None, clf_filename=None):
    print('Training classifiers...')
    # prepare the labels of localizer images (1401, 1410, 1421) and outcome images(4010, 4011, 4100, 4101, 4210, 4211)
    #  11311 11320 11331 12311 12320 12331
    # labels_loc = ((labels_loc%1000)/10).astype('int') # 40, 41, 42
    # labels_outcome = (labels_outcome/100).astype('int') #40, 41, 42
    accuracy_scores = np.zeros((1, len(times)))

    # train classifiers

    if trainingTimeIndex == None and trainingWindow == None:
        print('Time-point by time-point decoding..')

        for i in range(len(times)):
            clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(solver='liblinear')))
            # print('Parameters of the classsifier:')
            # print('C = ', clf.get_params('linearmodel__model__C'))
            # print('Penalty = ', clf.get_params('linearmodel__model__penalty'))

            # fit data
            clf.fit(data_loc[:, :, i], labels_loc)

            # predict the label of outcomes
            predictions = clf.predict(data_outcome[:, :, i])

            # check if all the perdictions are of one class
            testingIsAllSame(predictions)

            # evaluate performance
            acc = accuracy_score(labels_outcome, predictions)

            # check if all scores are zero
            testingIsAllZero(acc)

            # print('acc: ', acc)
            accuracy_scores[0, i] = acc
            del clf, acc, predictions

    elif trainingTimeIndex != None and trainingWindow == None:  # training at a time point
        print('Training time point: ', times[trainingTimeIndex])
        # classifier
        clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(solver='liblinear')))
        # fitting data
        clf.fit(data_loc[:, :, trainingTimeIndex], labels_loc)
        # predict the label of outcomes
        for i in range(len(times)):
            predictions = clf.predict(data_outcome[:, :, i])

            # check if all the perdictions are of one class
            testingIsAllSame(predictions)

            # evaluate performance
            acc = accuracy_score(labels_outcome, predictions)

            # check if all scores are zero
            testingIsAllZero(acc)

            accuracy_scores[0, i] = acc

            cm = confusion_matrix(labels_outcome, predictions)
            print('Confusion Matrix at: ', times[i])
            print(cm)
            del acc

        # save the model to disk
        pickle.dump(clf, open(clf_filename, 'wb'))

    elif trainingTimeIndex == None and trainingWindow != None:  # training using a window

        trainingWindow_size = trainingWindow[1] - trainingWindow[0] + 1

        # classifier
        clf = make_pipeline(StandardScaler(), LinearModel(LogisticRegression(solver='liblinear')))

        # TRAINING
        # prepare training data
        trainingData = np.zeros((data_loc.shape[0] * trainingWindow_size, data_loc.shape[1]), )
        trainingLabels = np.zeros((data_loc.shape[0] * trainingWindow_size), )
        data_ind = 0

        # Fill validation set with data from training time window
        for tp in range(trainingWindow[0], trainingWindow[1] + 1):
            trainingData[data_ind * data_loc.shape[0]:(data_ind + 1) * data_loc.shape[0], :] = data_loc[:, :, tp]  # data to test
            trainingLabels[data_ind * data_loc.shape[0]:(data_ind + 1) * data_loc.shape[0]] = labels_loc
            data_ind += 1


        # fitting data
        clf.fit(trainingData, trainingLabels)

        # TEST
        # predict the label of outcomes
        for i in range(len(times)):
            predictions = clf.predict(data_outcome[:, :, i])
            # check if all the perdictions are of one class
            testingIsAllSame(predictions)
            # evaluate performance
            acc = accuracy_score(labels_outcome, predictions)
            # check if all scores are zero
            testingIsAllZero(acc)
            accuracy_scores[0, i] = acc
            del acc
        # save the model to disk
        pickle.dump(clf, open(clf_filename, 'wb'))

    return accuracy_scores


def loadScores(sub_s_list, resultsPath, results_filename, isPermuted=False):
    results_group = []
    # Plot group-level localizer
    for s in range(len(sub_s_list)):
        if isPermuted == False: # --> load scores of trained classifiers

            results_filename_tmp = resultsPath +  'SingleSubjectResults/' + sub_s_list[s] + '/' + str(sub_s_list[s]) + results_filename
            results_tmp = np.load(results_filename_tmp)

            # check if all scores are zero
            testingIsAllZero(results_tmp)

            results_group.append(results_tmp)
            del results_tmp

        else:  # --> load scores of classifiers trained on permuted labels
            results_filename_tmp = resultsPath + 'SingleSubjectResults/' + sub_s_list[s] + '/Permutations/' + str(sub_s_list[s]) + results_filename
            results_subject = np.load(results_filename_tmp)  # shape = (1, ntimes)
            # check if all scores are zero
            testingIsAllZero(results_subject)

            results_group.append(results_subject)
            del results_subject


    results_group_arr = np.asarray(results_group)  # convert list to array

    return results_group_arr
