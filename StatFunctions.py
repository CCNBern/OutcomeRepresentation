
import numpy as np
from testingFunctions import testingType, testingIsAllSame, testingIsAllZero, testingIsEmpty

from scipy import stats as st
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.stats.anova import AnovaRM

from scipy.ndimage import measurements
import scipy
import pandas as pd
from mne.stats import f_mway_rm



def compareTwoConditions_wilcoxon(cond1, cond2, startTimeInd_forCorrecting, isAutocorrelation, endTimeInd_forCorrecting=None, correcting=True):

    p_vals_all, significant_timePoints = [], []


    if endTimeInd_forCorrecting == None:
        endTimeInd_forCorrecting = -1

    if np.array_equal(cond1, cond2) == False:

        p_thres = 0.05

        if len(cond1.shape) == 2: # 2d data

            nPoints = cond1.shape[1] # len ntimes
            nPoints_array = np.arange(nPoints)

            p_vals_all = np.ones(nPoints)

            if isAutocorrelation == True:
                loopStart = 1
            else:
                loopStart = 0

            for t in np.arange(loopStart, nPoints):
                #print('t: ', np.all(cond1[:, t] - cond2[:,t] == 0))
                x, p_vals_all[t] = wilcoxon(cond1[:, t], cond2[:, t], alternative='two-sided')

            #print('p values:  ', p_vals_all)
            #print('p values shape:  ', p_vals_all.shape)
            testingIsEmpty(p_vals_all)
            testingIsAllZero(p_vals_all)

            # correction for multiple comparisons
            if correcting  == True:
                p_vals_all_corrected_all = np.ones(p_vals_all.shape)
                # correct for multiple comparisons: only correct the ones after the plotting time start
                rejected, p_vals_all_corrected = fdrcorrection(p_vals_all[startTimeInd_forCorrecting:endTimeInd_forCorrecting],
                                                              alpha=0.1, method='indep', is_sorted=False)


                testingIsEmpty(p_vals_all_corrected)
                p_vals_all_corrected_all[startTimeInd_forCorrecting:endTimeInd_forCorrecting] = p_vals_all_corrected
                print('p values corrected: ', p_vals_all_corrected_all)

                significant_timePoints = nPoints_array[np.where(p_vals_all_corrected_all < p_thres)]
                print('points that are significantly different: ', significant_timePoints)

                return p_vals_all_corrected_all, significant_timePoints

            # No correction for multiple comparisons
            else:

                significant_timePoints = nPoints_array[np.where(p_vals_all < p_thres)]

                print('points that are significantly different: ', significant_timePoints)



        elif len(cond1.shape) == 1: # 1d data

            x, p_vals_all = wilcoxon(cond1, cond2, alternative='two-sided')
            print('p values: ', p_vals_all)
            significant_timePoints = None

    else:
        print('Two arrays are the same')


    return p_vals_all, significant_timePoints



def regression_f(model, outcome):

    #  compute f and p values for linear regression
    Y = outcome
    res = model.residuals_
    SS_res = np.sum(np.square(res))
    SS_tot = sum( (Y - np.mean(Y))**2 )
    SS_mod = SS_tot - SS_res
    df_mod = model.df_model_
    df_res = model.df_resid_
    MS_mod = SS_mod / df_mod
    MS_res = SS_res / df_res
    F = MS_mod / MS_res
    p = st.f.sf(F, df_mod, df_res)

    return F, p, df_mod, df_res
