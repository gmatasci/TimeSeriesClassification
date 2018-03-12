"""
Project Name: Time-series classification
Authors: Giona Matasci (giona.matasci@gmail.com)
File Name: TS_classif_main.py
Objective: Time-series classification using baselines methods and DL architechtures (CNN, RNN, etc.) on UCR Time Series Classification benchmark data
"""

## TODO:
# - sample uniformley across classes
# - add data augmentation
# - add pre-extracted temporal features as input to the CNN

# DONE:
# - test dropout after maxpooling vs after dense layer only -- done, added after every conv and dense layer
# - compare fit with default parameters (option #1) vs fit with manually specified step_per_epoch and val_steps (option #2) -- same thing, so removed the more verbose manual specification
# - check why class 0 doesnt get predicted -- problem when batch size is too big (> 32)
# - organize DL functions to separate common things -- managed to change dropout once model is already defined, so now model definition is before the bug function grid-searching for the best parameters
# - check plateau lr reduction -- works and one can see it with verbose=1. In TB the graph for lr has y-axis that is too big, but one can see lr varying thanks to ReduceLROnPlateau
# - write models and tb logs etc to dataset-specific folder -- done, so one can check training plots per dataset and
# - check whats wrong with EEG5000 -- the dataset was sorted with label 2 samples all at the end, shuffling solved the problem
# - test averagepooling vs dense layer -- global average pooling seems to perform better.

## IMPORT MODULES ------------------------------------------------------------------------------------------------------

import os
import sys
import glob
import shutil
import pickle
import time
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import json
import h5py
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score, classification_report, f1_score
import tensorflow as tf
from keras import backend as K
from keras import models, optimizers, callbacks, layers
from keras.utils import np_utils

from TS_DL_architectures import*

sys.path.insert(0, "C:/Projects/Python/utils/")
from utils import*

# To see tensor shape when hovering over name
np.set_string_function(lambda a: str(a.shape), repr=False)

## See devices to check whether GPU or CPU are being used
# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

## PARAMETERS -------------------------------------------------------------------------------------------

PARAMS = {}

PARAMS['verbose'] = True

# PARAMS['exp_name'] = r'4datasets'  # Experiment name, to create a separate folder with results, figures, Tensorboard logs, etc.
PARAMS['exp_name'] = r'debugging'  # to avoid overwriting files in experiment folders

PARAMS['run_training'] = True   # if True trains model, if False loads saved model and goes to evalaution on test set
# PARAMS['run_training'] = False

PARAMS['subsetting'] = False   # TODO subset datasets for testing purposes

# PARAMS['plot_figures'] = True
PARAMS['plot_figures'] = False

# PARAMS['datasets'] = ['UWaveGestureLibraryAll', 'PhalangesOutlinesCorrect', 'ElectricDevices', 'StarLightCurves']    # 'Adiac'
PARAMS['datasets'] = ['StarLightCurves']    # 'Adiac'

# PARAMS['methods'] = ['kNN', 'RF', 'CNN_gl_avg', 'CNN_fully_connected', 'CNN_LSTM']  # 'CNN_fully_connected' not working as expected
PARAMS['methods'] = ['CNN_gl_avg']  # 'CNN_fully_connected'

PARAMS['seed'] = 2018

PARAMS['normaliz'] = True
# PARAMS['normaliz'] = False

# Deep Learning-specific parameters (valid for all DL methods)
PARAMS['DL'] = {}
PARAMS['DL']['pct_val'] = 0.25   # fraction of samples to use as validation set
PARAMS['DL']['val_metric'] = 'val_loss'  # alternatively monitor='val_acc'
PARAMS['DL']['epochs'] = 2000
PARAMS['DL']['patience'] = 4  # number of epochs we tolerate the validation accuracy to be stagnant (not larger than current best accuracy), with patience for ReduceLROnPlateau set to np.round(PARAMS['DL']['patience']*0.8)
PARAMS['DL']['learn_rate'] = 1e-4   # 0.0001 usually gives the best results with Adam, and ReduceLROnPlateau will fine tune it if we get stuck in local minima

# DL hyperparamters to be tuned by grid
# PARAMS['DL']['HP'] = {'batch_size_trn': [4, 8, 16],
#                       'dropout': [0, 0.2, 0.4]}  # dropout 0 means we keep all the units
PARAMS['DL']['HP'] = {'batch_size_trn': [8],
                      'dropout': [0]}

# PARAMS['DL']['augmentation'] = True   # TODO

# CV parameters
PARAMS['nr_folds'] = 5

# kNN-specific parameter
PARAMS['kNN'] = {}
PARAMS['kNN']['k'] = list(range(1, 11)) + list(range(15, 31, 5))
# PARAMS['kNN']['k'] = [1]

# RF-specific parameters
PARAMS['RF'] = {}
# PARAMS['RF']['ntrees'] = [50, 1000]
PARAMS['RF']['ntrees'] = [1000]
PARAMS['RF']['mtry'] = ['sqrt', 'log2', 0.33]   # sqrt corresponds to R's default for classification, 0.33 to R's default for regression
# PARAMS['RF']['mtry'] = ['sqrt']   # sqrt corresponds to R's default for classification, 0.33 to R's default for regression

PARAMS['conf_mat_norm'] = False   # whether to normalize confusion by the true totals

## Definition of the global directories
PARAMS['dirs'] = {}
PARAMS['dirs']['base'] = r'C:\Projects\Trials\TimeSeriesClassif'
PARAMS['dirs']['fig_exploratory'] = os.path.join(PARAMS['dirs']['base'], 'wkg', PARAMS['exp_name'], 'Figures', 'Exploratory')
PARAMS['dirs']['res'] = os.path.join(PARAMS['dirs']['base'], 'wkg', PARAMS['exp_name'], "Results")
PARAMS['dirs']['data'] = r'C:\Projects\Trials\TimeSeriesClassif\Data\UCR_TS_Archive_2015'

## DEFINE FUNCTIONS ------------------------------------------------------------------

def readucr(filename):
    data = np.loadtxt(filename, delimiter = ',')
    Y = data[:,0]
    X = data[:,1:]
    return X, Y

def shuffle_rows(X, Y, seed=None):
    assert X.shape[0] == Y.shape[0]
    np.random.seed(seed=seed)
    p = np.random.permutation(X.shape[0])
    return X[p], Y[p]

def DL_grid_search(trn_data, val_metric, model_name, model, hparams_dict):
    """
    For a given set of hyperparameters train network and find best model by early stopping
    :param trn_data: training dataset, as an (X, Y) tuple
    :param val_metric: metric to use both for earlystopping and to select best hparams
    :param model_name: name to use when saving model files and logs
    :param model: Keras model object
    :param hparams_dict: dictionary with lists of values to test for batch_size and dropout
    :return best_model_path: path to best model (saved by the Checkpoint callback)
    :return best_hparams: best set of hparams to show in test results table
    :return grid_search_df: df with gridsearch results
    """

    X_trn, Y_trn = trn_data
    nr_classes = len(np.unique(Y_trn))

    Y_trn_one_hot = np_utils.to_categorical(Y_trn, nr_classes)

    # To have proper shape for temporal data 3D tensor: (batch_size, steps, features)
    X_trn = X_trn.reshape(X_trn.shape + (1,))

    if val_metric == 'val_loss':
        val_mode = 'min'
    elif val_metric == 'val_acc':
        val_mode = 'max'

    # Gridsearch over the hyperparameters
    grid_search_list = []  # list to be converted to pd dataframe
    for bs in hparams_dict['batch_size_trn']:
        for do in hparams_dict['dropout']:
            print('Batch size = %g, dropout = %g' % (bs, do))

            # Build string to give best models and folders different names for each hparams combination
            hparams_str = '%s_bs_%g_do_%g' % (model_name, bs, do)

            # Delete and recreate folder for Tensorboard logs
            log_dir_hparams = os.path.join(PARAMS['dirs']['log'], hparams_str)
            if os.path.exists(log_dir_hparams):
                shutil.rmtree(log_dir_hparams)
            os.makedirs(log_dir_hparams)

            # Earlystopping callback with a given patience
            earlystop_callback = callbacks.EarlyStopping(monitor=val_metric, mode=val_mode, patience=PARAMS['DL']['patience'])  # prefix 'val_' added automatically by Keras (based on name of Loss function)

            # Tensorboard callback to visualize network/evolution of metrics
            tb_callback = callbacks.TensorBoard(log_dir=log_dir_hparams, write_graph=True)

            # Checkpoint callback to save model each time the validation score (loss, acc, etc.) improves
            best_model_path = os.path.join(PARAMS['dirs']['model'], '%s.hdf5' % (hparams_str))
            checkpoint_callback = callbacks.ModelCheckpoint(best_model_path, monitor=val_metric, mode=val_mode, verbose=1, save_best_only=True)

            # Learning rate callback to reduce learning rate if val_loss does not improve after patience epochs (divide by 10 each time till a minimum of 0.000001)
            # patience value set to 80% of of the EarlyStopping patience to have ReduceLROnPlateau act first, then if nothing improves for another 20% of patience steps, we stop
            reduce_lr_callback = callbacks.ReduceLROnPlateau(monitor=val_metric, mode=val_mode, factor=0.1,
                                                             patience=np.ceil(PARAMS['DL']['patience']*0.8), min_lr=1e-6,
                                                             verbose=1)
            # Adapt dropout based on hp
            for layer in model.layers:
                if isinstance(layer, layers.Dropout):
                    layer.rate = do

            history = model.fit(x=X_trn, y=Y_trn_one_hot,
                                epochs=PARAMS['DL']['epochs'],
                                batch_size=bs,
                                validation_split=PARAMS['DL']['pct_val'],
                                callbacks=[reduce_lr_callback, checkpoint_callback, earlystop_callback, tb_callback],
                                verbose=2)

            # Get log and add epoch information
            log = pd.DataFrame(history.history)
            log['epoch'] = np.arange(log.shape[0])+1

            ## Get best metric value and corresponding epoch for current set of hparams
            if val_metric == 'val_loss':
                best_val_metric = log.loc[log[val_metric].idxmin][val_metric]
                best_epoch = log.loc[log[val_metric].idxmin]['epoch']
            elif val_metric == 'val_acc':
                best_val_metric = log.loc[log[val_metric].idxmax][val_metric]
                best_epoch = log.loc[log[val_metric].idxmax]['epoch']

            grid_search_list.append({'batch_size': bs, 'dropout': do,
                                     'epoch': best_epoch, 'val_score': best_val_metric,
                                     'model_path': best_model_path})  # fill row entries with dictionary

    # Get best values
    grid_search_df = pd.DataFrame(grid_search_list)  # convert to pd dataframe
    if PARAMS['DL']['val_metric'] == 'val_loss':
        grid_search_df.sort_values(by='val_score', ascending=True, inplace=True)
    elif PARAMS['DL']['val_metric'] == 'val_acc':
        grid_search_df.sort_values(by='val_score', ascending=False, inplace=True)

    best_model_path = grid_search_df['model_path'].iloc[0]
    best_hparams = grid_search_df.iloc[0].to_dict()
    best_hparams.pop('model_path')

    return best_model_path, best_hparams, grid_search_df

def predict_DL(model, X_tst, axis_labels):
    """
    Evaluates the model on the Validation images
    :param model: Keras model object to apply
    :param X_tst: Predictors (X) for the test set
    :return Y_tst_pred: column vector with predicted labels
    """

    X_tst = X_tst.reshape(X_tst.shape + (1,))

    nr_samples = X_tst.shape[0]

    print("Testing on %d samples" % (nr_samples))

    # Predict on test patches and convert to labels
    Y_tst_pred_tensor = model.predict(x=X_tst, verbose=1)   # 4D: nr_patches x height_out x width_out x nr_classes
    Y_tst_pred = np.argmax(Y_tst_pred_tensor, axis=axis_labels)   # 3D: nr_patches x height_out x width_out

    return Y_tst_pred

def assess_classif(Y, Y_pred, normalize_conf_mat=False, verbose=True):
    """
    :param Y: True labels
    :param Y_pred: Predicted labels
    :return RES: dictionary with the results on the test set: conf_mat (true labels as rows, predicted labels as columns), OA, Kappa, class_measures
    """

    # Assess test predictions and save results
    RES = {}
    conf_mat = confusion_matrix(Y, Y_pred)
    if normalize_conf_mat:
        RES['conf_mat'] = np.round((conf_mat.astype(np.float) / conf_mat.sum(axis=1)[:, np.newaxis])*100, 1)  # normalized by true labels totals (true labels as rows, predicted labels as columns)
    else:
        RES['conf_mat'] = conf_mat
    RES['OA'] = np.round(accuracy_score(Y, Y_pred)*100, 2)
    RES['Kappa'] = cohen_kappa_score(Y, Y_pred)
    RES['Mean_F1_score'] = f1_score(Y, Y_pred, average='macro')  # averaged individual class F1-scores. With average='macro': unweighted average, with average='weighted': weights proportional to support (the number of true instances for each label)
    RES['class_measures'] = classification_report(Y, Y_pred)

    if verbose:
        print('Classification results:\n\n '
              'Confusion matrix:\n %s \n\n '
              'OA=%.2f, Kappa=%.3f, Mean F1 score=%.3f \n\n '
              'Class-specific measures:\n %s'
              % (RES['conf_mat'], RES['OA'], RES['Kappa'], RES['Mean_F1_score'], RES['class_measures']))

    return RES


## START ---------------------------------------------------------------------

if __name__ == '__main__':

    print(python_info())

    print('TS_classif_main.py: started on %s' % time.strftime('%a, %d %b %Y %H:%M:%S', time.localtime()))

    start_time = tic()

    K.clear_session()  # release the memory on the GPU

## LOOP OVER DATASETS --------------------------------------------------------------------------------------------------

    RES = {}
    for dataset in PARAMS['datasets']:

        print('Dataset: %s' % dataset)

        # Define dataset-specific directories
        PARAMS['dirs']['log'] = os.path.join(PARAMS['dirs']['base'], 'wkg', PARAMS['exp_name'], 'Tensorboard_logs', dataset)  # to be added to 'dirs' dictionary only when dataset-specific folders will be created inside loop
        PARAMS['dirs']['model'] = os.path.join(PARAMS['dirs']['base'], 'wkg', PARAMS['exp_name'], 'Models', dataset)
        PARAMS['dirs']['best_model'] = os.path.join(PARAMS['dirs']['model'], 'Best_model')

        # Create all directories (the global ones will be created only in the 1st round of the loop and skipped afterwards)
        for name, dir in PARAMS['dirs'].items():
            if not os.path.exists(dir):
                os.makedirs(dir)

## PREPROCESSING --------------------------------------------------------------------------------------------------

        # Read data
        X_train_raw, Y_train_raw = readucr(os.path.join(PARAMS['dirs']['data'], dataset + '/' + dataset + '_TRAIN'))
        X_test_raw, Y_test_raw = readucr(os.path.join(PARAMS['dirs']['data'], dataset + '/' + dataset + '_TEST'))
        nr_classes = len(np.unique(Y_test_raw))

        # Shuffle rows of dataset (consistently across X and Y)
        X_train_raw, Y_train_raw = shuffle_rows(X_train_raw, Y_train_raw, seed=PARAMS['seed'])
        X_test_raw, Y_test_raw = shuffle_rows(X_test_raw, Y_test_raw, seed=PARAMS['seed'])

        # To have continuous labels from 0 to nr_classes - 1
        Y_train = (Y_train_raw - Y_train_raw.min()) / (Y_train_raw.max() - Y_train_raw.min()) * (nr_classes - 1)
        Y_test = (Y_test_raw - Y_test_raw.min()) / (Y_test_raw.max() - Y_test_raw.min()) * (nr_classes - 1)

        # Data normalization
        if PARAMS['normaliz']:
            # Compute stats on trn set
            X_train_mean = X_train_raw.mean()
            X_train_std = X_train_raw.std()

            # Apply same normalization on both the training set and test set
            X_train = (X_train_raw - X_train_mean) / (X_train_std)
            X_test = (X_test_raw - X_train_mean) / (X_train_std)
        else:
            X_train = X_train_raw
            X_test = X_test_raw

        nr_samples_trn = X_train.shape[0]
        nr_samples_tst = X_test.shape[0]

        train_data = (X_train, Y_train)

## PLOT EXPLORATORY FIGURES --------------------------------------------------------------------------------------------------

        # Plot quantiles per class over time (class signatures)
        if PARAMS['plot_figures']:

            base_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
            quantiles_to_plot = [.25, .5, 0.75]

            colors = [len(quantiles_to_plot) * [base_colors[i]] for i in range(nr_classes)]
            colors = sum(colors, [])  # to unlist nested lists
            lines = nr_classes * ['--', '-', '--']
            styles = [colors[i]+lines[i] for i in range(len(colors))]

            train_raw_df = pd.DataFrame(np.hstack((np.expand_dims(Y_train, 1), X_train_raw)))
            train_raw_df.rename(columns={0:'class'}, inplace=True)
            class_quantiles_raw_ts = train_raw_df.groupby(['class']).quantile(quantiles_to_plot).transpose()
            class_quantiles_raw_ts.plot(kind='line', grid=True, style=styles, title='Raw')
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.savefig(os.path.join(PARAMS['dirs']['fig_exploratory'], 'Class_TS_quantiles_raw_%s.pdf' % dataset),
                        dpi=400, bbox_inches='tight')
            plt.close()

            if PARAMS['normaliz']:
                train_df = pd.DataFrame(np.hstack((np.expand_dims(Y_train, 1), np.squeeze(X_train))))
                train_df.rename(columns={0:'class'}, inplace=True)
                class_quantiles_ts = train_df.groupby(['class']).quantile(quantiles_to_plot).transpose()
                class_quantiles_ts.plot(kind='line', grid=True, style=styles, title='Normalized')
                plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
                plt.savefig(os.path.join(PARAMS['dirs']['fig_exploratory'], 'Class_TS_quantiles_normalized_%s.pdf' % dataset),
                            dpi=400, bbox_inches='tight')
                plt.close()


## TRAIN AND TEST MODELS --------------------------------------------------------------------------------------------------

        RES[dataset] = {}   # dictionary containing the results for each dataset
        RES[dataset]['test_res_table'] = []  # list with summary of results to be converted to pd dataframe
        for method in PARAMS['methods']:

            print('\tMethod: %s' % method)

            if method == 'CNN_gl_avg':

                if PARAMS['run_training']:

                    # Create UNet model (model.summary() shows nr of trainable parameters)
                    model = TS_CNN().create_model(X_shape=(X_train.shape[1], 1),
                                                  nr_classes=nr_classes,
                                                  learn_rate=PARAMS['DL']['learn_rate'],
                                                  last_layer='gl_avg_pooling')  # X_shape is nr_features x 1, i.e., an horizontal vector of values at each time step (Input layer does not include the batch size)

                    best_model_path, best_hparams, grid_search = DL_grid_search(trn_data=train_data,
                                                                                val_metric=PARAMS['DL']['val_metric'],
                                                                                model=model, model_name=method,
                                                                                hparams_dict=PARAMS['DL']['HP'])

                else:

                    # If no training is performed, it means the best model is stored in PARAMS['dirs']['best_model']
                    # TODO to change
                    best_model_path = glob.glob(os.path.join(PARAMS['dirs']['best_model'], 'CNN_*'))[0]

                # Load best model from saved file, as model object after .fit() is a snapshot at the "best epoch + patience" point
                best_model = models.load_model(best_model_path)

                # Apply model on test set (columns are containing labels in case of time-series)
                Y_test_pred = predict_DL(best_model, X_tst=X_test, axis_labels=1)

            elif method == 'CNN_fully_connected':

                if PARAMS['run_training']:

                    # Create UNet model (model.summary() shows nr of trainable parameters)
                    model = TS_CNN().create_model(X_shape=(X_train.shape[1], 1),
                                                  nr_classes=nr_classes,
                                                  learn_rate=PARAMS['DL']['learn_rate'],
                                                  last_layer='fully_connected')

                    best_model_path, best_hparams, grid_search = DL_grid_search(trn_data=train_data,
                                                                                val_metric=PARAMS['DL']['val_metric'],
                                                                                model=model, model_name=method,
                                                                                hparams_dict=PARAMS['DL']['HP'])

                else:

                    # If no training is performed, it means the best model is stored in PARAMS['dirs']['best_model']
                    # TODO to change
                    best_model_path = glob.glob(os.path.join(PARAMS['dirs']['best_model'], 'CNN_*'))[0]

                # Load best model from saved file, as model object after .fit() is a snapshot at the "best epoch + patience" point
                best_model = models.load_model(best_model_path)

                # Apply model on test set (columns are containing labels in case of time-series)
                Y_test_pred = predict_DL(best_model, X_tst=X_test, axis_labels=1)

            elif method == 'CNN_LSTM':

                if PARAMS['run_training']:

                    # Create UNet model (model.summary() shows nr of trainable parameters)
                    model = TS_CNN().create_model(X_shape=(X_train.shape[1], 1),
                                                  nr_classes=nr_classes,
                                                  learn_rate=PARAMS['DL']['learn_rate'],
                                                  last_layer='LSTM')

                    best_model_path, best_hparams, grid_search = DL_grid_search(trn_data=train_data,
                                                                                val_metric=PARAMS['DL']['val_metric'],
                                                                                model=model, model_name=method,
                                                                                hparams_dict=PARAMS['DL']['HP'])

                else:

                    # If no training is performed, it means the best model is stored in PARAMS['dirs']['best_model']
                    # TODO to change
                    best_model_path = glob.glob(os.path.join(PARAMS['dirs']['best_model'], 'CNN_*'))[0]

                # Load best model from saved file, as model object after .fit() is a snapshot at the "best epoch + patience" point
                best_model = models.load_model(best_model_path)

                # Apply model on test set (columns are containing labels in case of time-series)
                Y_test_pred = predict_DL(best_model, X_tst=X_test, axis_labels=1)

            elif method == 'RF':

                rf_model_path = os.path.join(PARAMS['dirs']['best_model'], 'RF_best_model.hdf5')

                if PARAMS['run_training']:

                    # Train RF and save model
                    rf = RandomForestClassifier(random_state=PARAMS['seed'], n_jobs=-1)
                    param_grid = dict(n_estimators=PARAMS['RF']['ntrees'], max_features=PARAMS['RF']['mtry'])
                    grid_search = GridSearchCV(rf, param_grid, cv=PARAMS['nr_folds'], scoring='accuracy')  # n_jobs=-1 might interfere with the same paramter set for the classifier
                    grid_search.fit(X=X_train, y=Y_train)
                    rf_best = grid_search.best_estimator_
                    best_hparams = grid_search.best_params_
                    best_hparams['cv_score'] = grid_search.best_score_
                    with open(rf_model_path, 'wb') as f:
                        pickle.dump(rf_best, f)

                else:

                    # If not training is performed, we have to load the model
                    with open(rf_model_path, 'rb') as f:
                        rf_best = pickle.load(f)

                # Predict on test set
                Y_test_pred = rf_best.predict(X_test)

            elif method == 'kNN':

                knn_model_path = os.path.join(PARAMS['dirs']['best_model'], 'kNN_best_model.hdf5')

                if PARAMS['run_training']:

                    knn = KNeighborsClassifier(n_jobs=-1)
                    param_grid = dict(n_neighbors=PARAMS['kNN']['k'])
                    grid_search = GridSearchCV(knn, param_grid, cv=PARAMS['nr_folds'], scoring='accuracy')
                    grid_search.fit(X=X_train, y=Y_train)
                    knn_best = grid_search.best_estimator_
                    best_hparams = grid_search.best_params_
                    best_hparams['cv_score'] = grid_search.best_score_
                    with open(knn_model_path, 'wb') as f:
                        pickle.dump(knn_best, f)

                else:

                    # If not training is performed, we have to load the model
                    with open(knn_model_path, 'rb') as f:
                        knn_best = pickle.load(f)

                ## Predict on test set
                Y_test_pred = knn_best.predict(X_test)

            RES[dataset][method] = {}
            RES[dataset][method]['grid_search'] = grid_search

            # Save results in dictionary with an entry per method
            res_dict = assess_classif(Y_test, Y_test_pred, normalize_conf_mat=PARAMS['conf_mat_norm'], verbose=PARAMS['verbose'])
            RES[dataset][method]['test_results'] = res_dict
            RES[dataset]['test_res_table'].append({'Method': method,
                                                   'OA': res_dict['OA'],
                                                   'Kappa': res_dict['Kappa'],
                                                   'Mean_F1_score': res_dict['Mean_F1_score'],
                                                   'BestHParams': best_hparams})

        # Convert to df, sort values and rearrange columns
        RES[dataset]['test_res_table'] = pd.DataFrame(RES[dataset]['test_res_table'])  # convert to pd dataframe
        RES[dataset]['test_res_table'].sort_values(by='OA', ascending=False, inplace=True)
        RES[dataset]['test_res_table'] = RES[dataset]['test_res_table'][['Method', 'OA', 'Kappa', 'Mean_F1_score', 'BestHParams']]

        # Save results for this dataset in results folder in a binary pickle file (to avoid having to serialize arrays)
        res_filename = 'RES_%s.pkl' % (PARAMS['exp_name'])
        with open(os.path.join(PARAMS['dirs']['res'], res_filename), 'wb') as f:
            pickle.dump(RES, f)

    # Convert to list any possible np array for json.dump() to work
    for key, val in PARAMS.items():
        if isinstance(val, np.ndarray):
            PARAMS[key] = val.tolist()

    # Save parameters of this run in results folder in a understandable JSON (open in text editor)
    params_filename = 'PARAMS_%s.json' % PARAMS['exp_name']
    with open(os.path.join(PARAMS['dirs']['res'], params_filename), 'w') as fp:
        json.dump(PARAMS, fp)

    print('Total ' + toc(start_time))

    bla = 1

# Reload some old results dictionary
with open('C:/Projects/Trials/TimeSeriesClassif/wkg/4datasets/Results/RES_4datasets.pkl', 'rb') as f: # 'r' for reading; can be omitted
    RES_OLD = pickle.load(f)         # load file content as mydict


