# This is a sample Python script.
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os
# os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-10.2/lib64'
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import tensorflow as tf
import seaborn as sns
import load_data
import preprocess_data
import getmodel_chexnet
import save_model
import argparse
import time
import random as python_random

def test_config():
    print(f'tf version: {tf.__version__}')
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print(tf.config.list_physical_devices())


def preprocess_pipeline(traindir, trainpaths, trainclasses, valdir, valpaths, valclasses,  testdir, testpaths, testclasses, preprocessflag):
    if preprocessflag:
        traindata, trainlabels, valdata, vallabels, testdata, testlabels, testdata2, testlabels2 = preprocess_data.preprocess_all(traindir, trainpaths, trainclasses, valdir, valpaths, valclasses, testdir, testpaths, testclasses, testdir, testpaths, testclasses)
    else:  # tempfix
        traindata, trainlabels, valdata, vallabels, testdata, testlabels, testdata2, testlabels2 = preprocess_data.preprocess_all(traindir[:100],                                                                                                   trainpaths[:100],
                                                                                                          trainclasses[:100],
                                                                                                          valdir[:100],
                                                                                                          valpaths[:100],
                                                                                                          valclasses[:100],
                                                                                                          testdir[:100],
                                                                                                          testpaths[:100],
                                                                                                          testclasses[:100], testdir[:100], testpaths[:100],testclasses[:100])
    return traindata, trainlabels, valdata, vallabels, testdata, testlabels

def getsingledata(s, lungname, f, rightflagset):
    # TODO - Change this
    sourcedir = 'Paper_KFoldData_Finalv2'
    if (s == 'lung'):
        if(rightflagset):
            lungdir = 'LungSeg_'+'Right'
        else:
            lungdir= 'LungSeg_'+'Left'
    elif (s == 'spine'):
        if(rightflagset):
            lungdir = 'SpineSeg_'+'Right'
        else:
            lungdir= 'SpineSeg_'+'Left'
    elif (s == 'none'):
        lungdir = 'NoSeg'

    if(s == 'none'):
        traindir = 'TrainFinal_Fold'+str(f)+'_Balanced'+lungname
    else:
        traindir = 'TrainFinal_Fold' + str(f) + '_Balanced'

    basedir = os.path.join(sourcedir, lungdir)
    traindir_balanced = os.path.join(basedir, traindir)
    testdir = 'TestFinal_Fold' + str(f)
    testdir_balanced = os.path.join(basedir, testdir)
    valdir = 'ValFinal_Fold' + str(f)
    valdir_balanced = os.path.join(basedir, valdir)

    return traindir_balanced, testdir_balanced, valdir_balanced, traindir_balanced, testdir_balanced, valdir_balanced


def getsingledata_master(s, lungnae, f, rightflagset):
    # TODO - Change this
    sourcedir = 'Paper_KFoldData_Finalv2'
    if (s == 'lung'):
        if (rightflagset):
            lungdir = 'LungSeg_' + 'Right'
        else:
            lungdir = 'LungSeg_' + 'Left'
    elif (s == 'spine'):
        if (rightflagset):
            lungdir = 'SpineSeg_' + 'Right'
        else:
            lungdir = 'SpineSeg_' + 'Left'
    elif (s == 'none'):
        lungdir = 'NoSeg'

    if (s == 'none'):
        traindir = 'TrainFinal_Fold' + str(f) + '_Balanced' + lungname
    else:
        traindir = 'TrainFinal_Fold' + str(f) + '_Balanced'

    basedir = os.path.join(sourcedir, lungdir)
    traindir_balanced = os.path.join(basedir, traindir)
    testdir = 'TestFinal_Fold' + str(f)+'_Master'
    testdir_balanced = os.path.join(basedir, testdir)
    valdir = 'ValFinal_Fold' + str(f)+'_Master'
    valdir_balanced = os.path.join(basedir, valdir)

    return traindir_balanced, testdir_balanced, valdir_balanced, traindir_balanced, testdir_balanced, valdir_balanced


def getdoubledata(s, lungname, f, rightflagset):
    sourcedir = 'Paper_KFoldData_Finalv2'
    if (s == 'lung'):
        if(rightflagset):
            lungdir = 'LungSeg_'+'Right'
        else:
            lungdir= 'LungSeg_'+'Left'
    elif (s == 'spine'):
        if(rightflagset):
            lungdir = 'SpineSeg_'+'Right'
        else:
            lungdir= 'SpineSeg_'+'Left'
    elif (s == 'none'):
        lungdir = 'NoSeg'

    if(s == 'none'):
        traindir1 = 'TrainFinal_Fold'+str(f)+'_Balanced'+lungname
        traindir2 = 'TrainFinal_Fold' + str(f)
    else:
        traindir1 = 'TrainFinal_Fold' + str(f) + '_Balanced'
        traindir2 = 'TrainFinal_Fold'+str(f)

    basedir = os.path.join(sourcedir, lungdir)
    traindir_balanced = os.path.join(basedir, traindir1)
    traindir_unbalanced = os.path.join(basedir, traindir2)

    testdir = 'TestFinal_Fold' + str(f)
    testdir_balanced = os.path.join(basedir, testdir)
    valdir = 'ValFinal_Fold' + str(f)
    valdir_balanced = os.path.join(basedir, valdir)

    return traindir_balanced, testdir_balanced, valdir_balanced, traindir_unbalanced, testdir_balanced, valdir_balanced


def getsmotedata(s,lungname,f,rightflagset):
    # TODO - Insert your source directory here
    sourcedir = 'RadiologyPaper_June2022//Paper_KFoldData_Finalv2'
    if (s == 'lung'):
        if (rightflagset):
            lungdir = 'LungSeg_' + 'Right'
        else:
            lungdir = 'LungSeg_' + 'Left'
    elif (s == 'spine'):
        if (rightflagset):
            lungdir = 'SpineSeg_' + 'Right'
        else:
            lungdir = 'SpineSeg_' + 'Left'
    elif (s == 'none'):
        lungdir = 'NoSeg'

    traindir = 'TrainFinal_Fold'+str(f)+'_Augmented'+lungname+'_Final'

    basedir = os.path.join(sourcedir, lungdir)
    traindir_balanced = os.path.join(basedir, traindir)
    testdir = 'TestFinal_Fold' + str(f)
    testdir_balanced = os.path.join(basedir, testdir)
    valdir = 'ValFinal_Fold' + str(f)
    valdir_balanced = os.path.join(basedir, valdir)


# Press the green button in the gutter to run the script.

def getvals(args):
    segment = args.segment
    foldval = args.fold
    sampling = args.sampling
    lungname = args.lung
    if(lungname=='left'):
        rightflagset = False
    else:
        rightflagset = True

    if segment=='all':
        allsegments = ['lung', 'spine', 'none']
    else:
        allsegments = []
        allsegments.append(segment)
    if foldval=='all':
        allfolds = [1, 2, 3, 4, 5]
    else:
        allfolds = []
        allfolds.append(foldval)

    return segment, foldval, sampling, lungname, rightflagset, allsegments, allfolds

if __name__ == '__main__':
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(108)

    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    python_random.seed(108)

    # The below set_seed() will make random number generation
    # in the TensorFlow backend have a well-defined initial state.
    # For further details, see:
    # https://www.tensorflow.org/api_docs/python/tf/random/set_seed
    tf.random.set_seed(108)

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    #tf.random.set_seed(0)
    modelname = 'CHEXNET'
    modifiedflag = False
    outputdir = 'ForPaperFinalModels_June2022_Master'

    # Presets for testing with minimum data
    preprocessflag = 1
    datagenerateflag = 1
    modelflag = 1

    # Parse the argument
    # Create the parser
    print('Creating parser....')
    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--lung', type=str, required=True)
    parser.add_argument('--sampling', type=str, required=True)
    parser.add_argument('--fold', type=str, required=True)
    parser.add_argument('--segment', type=str, required=True)
    args = parser.parse_args()
    # Print "Hello" + the user input argument
    print('Lung,', args.lung, ' Sampling ',args.sampling, ' Fold ',args.fold, ' Segment', args.segment)

    test_config()  # If this gives an error that means that there is a memory issue

    ###################################################################################################
    ###################### 1. Set presets, Make directory for model ###################################
    segment, foldval, sampling, lungname, rightflagset, allsegments, allfolds = getvals(args)
    print('Segment is ',allsegments)
    print('Lung name is ',lungname)
    ###################################################################################################
    ####################### 2. Preprocess data #############################################
    for s in allsegments:
        for f in allfolds:
            start = time.process_time()
            if(sampling=='under'):
                traindir_balanced, testdir_balanced, valdir_balanced, traindir_unbalanced, testdir_unbalanced, valdir_unbalanced = \
                    getsingledata(s, lungname, f, rightflagset) # TODO - change
            elif(sampling=='double'):
                traindir_balanced, testdir_balanced, valdir_balanced, traindir_unbalanced, testdir_unbalanced, valdir_unbalanced = \
                    getdoubledata(s, lungname, f, rightflagset) # TODO - change
            elif(sampling=='over'):
                traindir_balanced, testdir_balanced, valdir_balanced, traindir_unbalanced, testdir_unbalanced, valdir_unbalanced = \
                    getsmotedata(s, lungname, f, rightflagset)

            basemodel = 'model_' +modelname + '_' + lungname + '_' + s + '_' + sampling
            basemodeldir = os.path.join(outputdir,basemodel)
            os.makedirs(basemodeldir, exist_ok=True)
            modeldir = os.path.join(basemodeldir,'Fold'+str(f))
            os.makedirs(modeldir, exist_ok=True)
            print('model directory is ', modeldir)

            # LOAD DATA
            trainpaths_balanced, trainclasses_balanced, valpaths_balanced, valclasses_balanced, testpaths_balanced, \
            testclasses_balanced, testpaths_unbalanced, testclasses_unbalanced = load_data.load_imagepaths(
                traindir_balanced, valdir_balanced, testdir_balanced, testdir_unbalanced, rightflag=rightflagset)

            if (sampling == 'double'):
                trainpaths_unbalanced, trainclasses_unbalanced, valpaths_unbalanced, valclasses_unbalanced, testpaths_unbalanced, \
                testclasses_unbalanced, testpaths_unbalanced, testclasses_unbalanced = load_data.load_imagepaths(
                    traindir_unbalanced, valdir_unbalanced, testdir_unbalanced, testdir_unbalanced, rightflag=rightflagset)

            # PREPROCESS DATA
            traindata_balanced, trainlabels_balanced, valdata_balanced, vallabels_balanced, testdata_balanced, \
            testlabels_balanced = preprocess_pipeline( traindir_balanced, trainpaths_balanced, trainclasses_balanced,
                valdir_balanced, valpaths_balanced, valclasses_balanced, testdir_balanced, testpaths_balanced, testclasses_balanced, preprocessflag)

            if(sampling=='double'):
                traindata_unbalanced, trainlabels_unbalanced, valdata_unbalanced, vallabels_unbalanced, testdata_unbalanced, \
                testlabels_unbalanced = preprocess_pipeline(traindir_unbalanced, trainpaths_unbalanced, trainclasses_unbalanced,
                                                          valdir_unbalanced, valpaths_unbalanced, valclasses_unbalanced,
                                                          testdir_unbalanced, testpaths_unbalanced, testclasses_unbalanced,
                                                          preprocessflag)


            # ###########################################################################################################
            # ####################### 3. Get train, test data and run model #####################################
            #
            if datagenerateflag:
                trainX_balanced, trainY_balanced = preprocess_data.get_trainval(traindata_balanced, trainlabels_balanced)
                valX_balanced, valY_balanced = preprocess_data.get_trainval(valdata_balanced, vallabels_balanced)
                testX_balanced, testY_balanced = preprocess_data.get_trainval(testdata_balanced, testlabels_balanced)
                if (sampling == 'double'):
                    trainX_unbalanced, trainY_unbalanced = preprocess_data.get_trainval(traindata_unbalanced,
                                                                                        trainlabels_unbalanced)
                else:
                    trainX_unbalanced = trainX_balanced
                    trainY_unbalanced = trainY_balanced
                valX_unbalanced = valX_balanced
                valY_unbalanced = valY_balanced
                testX_unbalanced = testX_balanced
                testY_unbalanced = testY_balanced
            else:
                trainX_balanced, trainY_balanced = preprocess_data.get_trainval(traindata_balanced[:100], trainlabels_balanced[:100])
                valX_balanced, valY_balanced = preprocess_data.get_trainval(valdata_balanced[:100], vallabels_balanced[:100])
                testX_balanced, testY_balanced = preprocess_data.get_trainval(testdata_balanced[:100], testlabels_balanced[:100])
                print(trainX_balanced.shape)
                if(sampling=='double'):
                    trainX_unbalanced, trainY_unbalanced = preprocess_data.get_trainval(traindata_unbalanced[:100],
                                                                                    trainlabels_unbalanced[:100])
                else:
                    trainX_unbalanced = trainX_balanced
                    trainY_unbalanced = trainY_balanced
                valX_unbalanced = valX_balanced
                valY_unbalanced = valY_balanced
                testX_unbalanced = testX_balanced
                testY_unbalanced = testY_balanced

            if (modelflag):
                if(modifiedflag==True):
                    model, model_noft = getmodel_chexnet.build_model_doublelayer_modified(trainX_balanced, trainY_balanced,
                                                                                 valX_balanced, valY_balanced, testX_balanced,
                                                                                 testY_balanced, trainX_unbalanced,
                                                                                 trainY_unbalanced, valX_unbalanced,
                                                                                 valY_unbalanced,
                                                                                 testX_unbalanced, testY_unbalanced,
                                                                                 classNames=['0', '1', '2', '3'], len_classes=4,
                                                                                 loss0='categorical_crossentropy',
                                                                                 finetune=True, num_epochs_base=20,
                                                                                 num_epochs_finetune=20, earlystopflag=True,
                                                                                 outpath=modeldir)
                    # serialize model to JSON
                    model_json = save_model.save_modelfile(model, modelname=os.path.join(modeldir, 'model_' + modelname + '_' + lungname + '_' + s + '_' + sampling))

                else:
                    model, model_noft = getmodel_chexnet.build_model_doublelayer(trainX_balanced, trainY_balanced,
                                                                                     valX_balanced, valY_balanced,
                                                                                     testX_balanced,
                                                                                     testY_balanced, trainX_unbalanced,
                                                                                     trainY_unbalanced, valX_unbalanced,
                                                                                     valY_unbalanced,
                                                                                     testX_unbalanced, testY_unbalanced,
                                                                                     classNames=['0', '1', '2', '3'],
                                                                                     len_classes=4,
                                                                                     loss0='categorical_crossentropy',
                                                                                     finetune=True, num_epochs_base=40,
                                                                                     num_epochs_finetune=40,
                                                                                     earlystopflag=True,
                                                                                     outpath=modeldir)
                    # serialize model to JSON
                    model_json = save_model.save_modelfile(model, modelname=os.path.join(modeldir,
                                                                                         'model_' + modelname + '_' + lungname + '_' + s + '_' + sampling))

                time_taken = time.process_time() - start
                logstr = '\n' + 'ModelDir is ' + modeldir
                logstr = logstr + '\n' + 'TrainDir Balanced is ' + traindir_balanced
                logstr = logstr + '\n' + 'TrainDir UnBalanced is ' + traindir_unbalanced
                logstr = logstr + '\n' + 'TestDir is ' + testdir_balanced
                logstr = logstr + '\n' + 'ValDir is ' + valdir_balanced
                logstr = logstr + '\n' + 'TimeTaken is ' + str(time_taken)
                text_file = open(os.path.join(modeldir,"log.txt"), "w")
                n = text_file.write(logstr)
                text_file.close()
            else:
                model, model_noft = getmodel_chexnet.build_model_doublelayer(trainX_balanced, trainY_balanced,
                                                                             valX_balanced, valY_balanced, testX_balanced,
                                                                             testY_balanced, trainX_unbalanced,
                                                                             trainY_unbalanced, valX_unbalanced,
                                                                             valY_unbalanced,
                                                                             testX_unbalanced, testY_unbalanced,
                                                                             classNames=['0', '1', '2', '3'], len_classes=4,
                                                                             loss0='categorical_crossentropy',
                                                                             finetune=True, num_epochs_base=1,
                                                                             num_epochs_finetune=1, outpath=modeldir)

