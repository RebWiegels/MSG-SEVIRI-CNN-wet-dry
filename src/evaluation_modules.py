import numpy as np
import pandas as pd
from tqdm import tqdm
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

def Roc_curve(y_pred, y_true, negative=False, quickrun=False):
    '''
    Calculates x, y values (FPR, TPR) or if negative (FNR, TNR) to create ROC curve.
    --------
    y_true: bool
       ground truth
    y_pred: float
       predicted class probability with values in [0,1]
    negative: Boolean
        if True, Ratio FNR to TNR; if False, returns FPR to TPR
    quickrun: Boolean
        if True, only every 100s value of np.unique is calculated
        if False, every np.unique value
    ---------
    Returns
    ---------
    roc_list: array of shape (2, points) of floats
        each row of array represents on value of ROC curve
    '''
    roc_list = []
    if quickrun:
        for t in tqdm(np.unique(y_pred)[::100]):
            y_predicted=np.ravel(y_pred>t)  
            true_pos = np.sum(np.logical_and(y_true==1, y_predicted==1))
            true_neg = np.sum(np.logical_and(y_true==0, y_predicted==0))
            false_pos = np.sum(np.logical_and(y_true==0, y_predicted==1))
            false_neg = np.sum(np.logical_and(y_true==1, y_predicted==0))
            cond_neg = true_neg+false_pos
            cond_pos = true_pos+false_neg
            if negative:
                roc_list.append([true_neg/cond_neg, false_neg/cond_pos])
            else:
                roc_list.append([false_pos/cond_neg, true_pos/cond_pos])
    else:
        for t in tqdm(np.unique(y_pred)):
            y_predicted = np.ravel(y_pred>t)  
            true_pos = np.sum(np.logical_and(y_true==1, y_predicted==1))
            true_neg = np.sum(np.logical_and(y_true==0, y_predicted==0))
            false_pos = np.sum(np.logical_and(y_true==0, y_predicted==1))
            false_neg = np.sum(np.logical_and(y_true==1, y_predicted==0))
            cond_neg = true_neg + false_pos
            cond_pos = true_pos + false_neg
            if negative:
                roc_list.append([true_neg/cond_neg, false_neg/cond_pos])
            else:
                roc_list.append([false_pos/cond_neg, true_pos/cond_pos])

    return np.array(roc_list)

def confusion(y_pred, y_true, threshold, show=False):
    '''
    calculates confusion parameter and scores of certain threshold.
    ---------
    y_pred: array of 0 or 1
        predicted class 
    y_true: array of 0 or 1
        ground thruth
    threshold: float
        threshold in [0, 1]
    show: bool
        if True, will show figure of confusion matrix and values
    --------
    Returns
    --------
    ACC, MCC, PPV, NPV, Sensitivity, Specificity, FP, FN, TP, TN
    '''
    contingency = pd.crosstab(index=y_pred, columns=y_true, margins=False).reindex([1,0])[[1,0]] # set margins=True to add row/column of subtotals
    # evaluation
    contingency_abs =[[contingency.iloc[0, 0], contingency.iloc[0, 1]], [contingency.iloc[1, 0], contingency.iloc[1, 1]]]
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    oa = (contingency.iloc[ 0, 0] + contingency.iloc[ 1, 1])/ (contingency.iloc[ 0, 0] + contingency.iloc[ 0, 1] + contingency.iloc[ 1, 0] + contingency.iloc[ 1, 1])
    ppv = contingency.iloc[ 0, 0] / (contingency.iloc[ 0, 1] + contingency.iloc[ 0, 0])
    sensitivity = contingency.iloc[ 0, 0] / (contingency.iloc[1, 0] + contingency.iloc[ 0, 0])
    npv = contingency.iloc[ 1, 1] / (contingency.iloc[ 1, 0] + contingency.iloc[ 1, 1])
    specificity = contingency.iloc[ 1, 1] / (contingency.iloc[ 0, 1] + contingency.iloc[ 1, 1])
    if show:
        # plot matrix as heatmap
        fig = plt.figure(num=None, figsize=(4, 3), dpi=80, facecolor='w', edgecolor='k')
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        res = sns.heatmap(contingency, annot=True, fmt='.2f', cmap="YlGnBu")
        plt.title('Contingency table',fontsize=12)
        plt.show()
        print('Set thresholds: pc=', threshold*100, '%; rainfall amount of station=', threshold_station, 'mm/h')
        print('Overall accuaracy (night):', oa)
        print ('Positive predictive value (night):', ppv)
        print('Negative predictive value (night):', npv)
        print('Sensitivity (night):', sensitivity)
        print('Specificity (night):', specificity)
    return oa, mcc, ppv, npv, sensitivity, specificity, np.asarray(contingency_abs)

def conf_parameter(y_pred, y_true, th=[0.1, 0.2, 0.3, 0.4, 0.5]):
    '''
    Calculate ACC, MCC, FP, FN, TN, TP of thresholds as defined.
    ---------
    y_pred: numpy array
            predicted 
    y_true: numpy array
            true 
    th: array of float
            thresholds to calculate parameter
    ---------
    Returns
    ---------
    df: dataframe of scores
            number of rows depend of number of thresholds defined in th.
    
    '''

    ls_ac = []
    ls_mcc = []
    ls_FP = []
    ls_FN = []
    ls_TN = []
    ls_TP = []
    for i in th:
        t=i
        y_predicted=np.where(y_pred>t, 1, 0)  
        oa, mcc, ppv, npv, sensitivity, specificity, confusion_abs = confusion(y_predicted, y_true, threshold=t, show=False)
        ls_ac.append(oa)
        ls_mcc.append(mcc)
        ls_FP.append(confusion_abs[0, 1])
        ls_TP.append(confusion_abs[0, 0])
        ls_FN.append(confusion_abs[1, 0])
        ls_TN.append(confusion_abs[1, 1])
    df_results = pd.DataFrame({'threshold': th, 'oa': ls_ac, 'mcc': ls_mcc, 'TP':ls_TP,
                           'FP': ls_FP, 'FN': ls_FN, 'TN': ls_TN})
    return df_results


def confusion_matrix (y_pred, y_true):
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_true, y_pred).ravel()
    oa = (tp + tn) / (tp + tn + fp +  fn)
    mcc = metrics.matthews_corrcoef(y_true, y_pred)
    sens = tp / (tp + fn)
    spec = tn / (tn + fp)
    f_sc = (2 * tp) / (2 * tp + fp + fn)
    return oa, mcc, sens, spec, tn, fp, fn, tp