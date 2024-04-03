import numpy as np
import pandas as pd
import pingouin as pg

from StatFunctions import regression_f
from PlotFunctions import plotRegression
from PSDAnalysis_functions import calculateDiffMetric

seed = 42


def linkingWithBehaviour(metricType, data_ft_group, data_ft_group_perm, freqs, isAbs, cond, base_path_beh,
                         split_num, resultsPath, usePartialData, sigPoints, isLessLikely=False, typeLesslikely=None,
                         correcting=None, sub_s_list=None):

    # calculate the area between real and perm curves
    metric_delta_arr = calculateDiffMetric(metricType, data_ft_group, data_ft_group_perm, freqs, isAbs, sigPoints)

    # load behavioural accuracy
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

    # create dataframe for plotting
    df_toPlot = pd.DataFrame(data_toPlot)
    df_toPlot['subjects'] = sub_s_list

    if 'cd' in cond:
        color = '#C40E94'
    elif 'cf' in cond:
        color = '#008A09'
    else:
        color = 'black'

    # separate dependent (y) and independent variables (x)
    x_data = None
    y_data = None

    if 'cd' in cond:
        x_data = data_toPlot[metricType + "_cd"]
        y_data = data_toPlot['Accuracy_cd']

    elif 'cf' in cond:
        x_data = data_toPlot[metricType + "_cf"]
        y_data = data_toPlot['Accuracy_cf']

    if y_data.all() != None:


        # bulid regression model
        model = pg.linear_regression(x_data, y_data)

        # compute f and p values
        fval, pval, df_model, df_residual = regression_f(model, y_data)
        print('F value: ', fval)
        print('P value: ', pval)
        print('df_model: ', df_model)
        print('df_residual: ', df_residual)

        # save model outputs and stats
        if isLessLikely == False: # analysis of most likely
            model_filename = resultsPath + 'Abs_regression_PSDVSbehaviour_' + cond + '_linreg_model'

            if correcting == True:
                model_filename += '_correctedPvals'
            else:
                model_filename += '_uncorrectedPvals'


            model_filename += '.txt'

        else: # analysis of lessLikely
            model_filename = resultsPath + 'Abs_regression_PSDVSbehaviour_' + cond + '_lessLikely_' + typeLesslikely + '_linreg_model'

            if correcting == True:
                model_filename += '_correctedPvals'
            else:
                model_filename += '_uncorrectedPvals'

            model_filename += '.txt'



        with open(model_filename, 'w') as f:
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_colwidth', -1)

            f.write('condition: ' + cond + '\n')
            f.write('Model: \n' + str(model) + '\n')
            f.write('F value: ' + str(fval) + ', P  value: ' + str(pval) + '\n')
            f.write('df model: ' + str(df_model) + ', df residual: ' + str(df_residual) + '\n')

        # plot
        plotRegression(model, fval, pval, color, df_toPlot, cond, resultsPath, usePartialData, split_num,
                       typeOfAnalysis='PSD',metricType=metricType, isLessLikely=isLessLikely,
                       typeLesslikely=typeLesslikely, correcting=correcting)
