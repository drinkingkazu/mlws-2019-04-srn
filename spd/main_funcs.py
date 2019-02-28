from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import datetime
import glob
import sys
import numpy as np
from spd.iotools import DataLoaderFactory
from spd.trainval import trainval
import spd.utils as utils
import torch

def iotest(flags):
    # IO configuration
    dataloader = DataLoaderFactory(flags)
    t0=time.time()
    for i, sample in enumerate(dataloader):
        data,label,index = sample
        t1=time.time()
        print('Batch',i,'time',t1-t0,'[s]')
        print(' ... index:',index)
        assert len(label) == BATCH_SIZE
        t0=time.time()
        if i==100: break

class Handlers:
    trainer      = None
    dataloader   = None
    csv_logger   = None
    train_logger = None
    iteration    = 0

def train(flags):
    flags.TRAIN = True
    handlers = prepare(flags)
    train_loop(flags, handlers)

def inference(flags):
    flags.TRAIN = False
    handlers = prepare(flags)
    inference_loop(flags, handlers)

def prepare(flags):
    if len(flags.GPUS) > 0:
        torch.cuda.set_device(flags.GPUS[0])
    handlers = Handlers()

    # IO configuration
    handlers.dataloader = DataLoaderFactory(flags)

    # Trainval configuration
    handlers.trainer = trainval(flags)

    # Restore weights if necessary
    loaded_iteration = handlers.trainer.initialize()
    handlers.iteration = (loaded_iteration if flags.TRAIN else 0)

    # Weight save directory
    if flags.WEIGHT_PREFIX:
        save_dir = flags.WEIGHT_PREFIX[0:flags.WEIGHT_PREFIX.rfind('/')]
        if save_dir and not os.path.isdir(save_dir): os.makedirs(save_dir)
        
    # Log save directory
    if flags.LOG_DIR:
        if not os.path.exists(flags.LOG_DIR): os.mkdir(flags.LOG_DIR)
        logname = '%s/train_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration)
        if not flags.TRAIN:
            logname = '%s/inference_log-%07d.csv' % (flags.LOG_DIR, loaded_iteration)
        handlers.csv_logger = utils.CSVData(logname)
    return handlers

def log(handlers, tstamp_iteration, tspent_iteration, tsum, res, flags, epoch):

    report_step  = flags.REPORT_STEP and ((handlers.iteration+1) % flags.REPORT_STEP == 0)

    loss = np.mean(res['loss'])
    acc  = np.mean(res['accuracy'])

    if len(flags.GPUS) > 0:
        mem = utils.round_decimals(torch.cuda.max_memory_allocated()/1.e9, 3)
    else:
        mem = -1

    # Report (logger)
    if handlers.csv_logger:
        handlers.csv_logger.record(('iter', 'epoch', 'titer', 'tsumiter'),
                                   (handlers.iteration,epoch,tspent_iteration,tsum))
        #handlers.csv_logger.record(('tio', 'tsumio'),
        #                           (handlers.data_io.tspent_io,handlers.data_io.tspent_sum_io))
        handlers.csv_logger.record(('gpumem', ), (mem, ))
        tmap, tsum_map = handlers.trainer.tspent, handlers.trainer.tspent_sum
        if flags.TRAIN:
            handlers.csv_logger.record(('ttrain','tsave','tsumtrain','tsumsave'),
                                       (tmap['train'],tmap['save'],tsum_map['train'],tsum_map['save']))
        # else:
        handlers.csv_logger.record(('tforward','tsave','tsumforward','tsumsave'),
                                   (tmap['forward'],tmap['save'],tsum_map['forward'],tsum_map['save']))

        handlers.csv_logger.record(('loss','acc'),(loss,acc))
        handlers.csv_logger.write()

    # Report (stdout)
    if report_step:
        loss = utils.round_decimals(loss,   4)
        tmap  = handlers.trainer.tspent
        tfrac = utils.round_decimals(tmap['train']/tspent_iteration*100., 2)
        tabs  = utils.round_decimals(tmap['train'], 3)
        epoch = utils.round_decimals(epoch, 2)

        if flags.TRAIN:
            msg = 'Iter. %d (epoch %g) @ %s ... train time %g%% (%g [s]) GPU mem. %g GB \n'
            msg = msg % (handlers.iteration, epoch, tstamp_iteration, tfrac, tabs, mem)
        else:
            msg = 'Iter. %d (epoch %g) @ %s ... forward time %g%% (%g [s]) GPU mem. %g GB \n'
            msg = msg % (handlers.iteration, epoch, tstamp_iteration, tfrac, tabs, mem)
        msg += '   Segmentation: loss %g accuracy %g\n' % (loss, acc)
        print(msg)
        sys.stdout.flush()
        if handlers.csv_logger: handlers.csv_logger.flush()
        if handlers.train_logger: handlers.train_logger.flush()

def get_data_minibatched(handlers, flags):
    """
    Handles minibatching the data
    """
    blob_array=[]
    #    blob_array = ([#MINIBATCH,#GPU,DATA],[MINIBATCH,#GPU,LABEL])
    t=0
    t0=time.time()
    ctr = flags.BATCH_SIZE / flags.MINIBATCH_SIZE
    while ctr > 0:
        blob={'data':[],'label':[],'index':[]}
        procs = max(1,len(flags.GPUS))
        tio0=time.time()
        for i, sample in enumerate(handlers.dataloader):
            t += (time.time() - tio0)
            data,label,index = sample
            blob['data' ].append(data)
            blob['label'].append(label)
            blob['index'].append(index)
            tio0=time.time()
            print('appended',tio0)
            if (i+1) == procs: break
        blob_array.append(blob)
        ctr -= 1
    t1 = time.time()
    print('blob prep',t1-t0)
    print('blob read',t)
    return blob_array

def train_loop(flags,handlers):

    tsum=0.
    blob={'data':[],'label':[],'index':[]}
    tstart = time.time()
    
    minibatch_ctr = 0
    tstart_iteration = time.time()
    while handlers.iteration < flags.ITERATION:

        for sample in handlers.dataloader:
            data,label,index = sample
            blob['data'].append(data)
            blob['label'].append(label)
            blob['index'].append(index)
            minibatch_ctr += 1
                
            if minibatch_ctr % (flags.BATCH_SIZE/flags.MINIBATCH_SIZE) > 0: continue
                
            epoch = handlers.iteration * float(flags.BATCH_SIZE) / (len(handlers.dataloader.dataset))
            tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            checkpt_step = flags.CHECKPOINT_STEP and flags.WEIGHT_PREFIX and ((handlers.iteration+1) % flags.CHECKPOINT_STEP == 0)

            # Train step
            res = handlers.trainer.train_step(blob, batch_size=flags.BATCH_SIZE)
        
            # Save snapshot
            if checkpt_step:
                handlers.trainer.save_state(handlers.iteration)

            tspent_iteration = time.time() - tstart_iteration
            tsum += tspent_iteration
            log(handlers, tstamp_iteration, tspent_iteration, tsum, res, flags, epoch)

            # Increment iteration counter
            handlers.iteration += 1        
            if handlers.iteration >= flags.ITERATION:
                break
            tstart_iteration = time.time()
            blob={'data':[],'label':[],'index':[]}
    # Finalize
    if handlers.csv_logger:
        handlers.csv_logger.close()

def inference_loop(flags,handlers):

    tsum=0.
    blob={'data':[],'label':[],'index':[]}
    tstart = time.time()
    
    minibatch_ctr = 0
    tstart_iteration = time.time()
    while handlers.iteration < flags.ITERATION:

        for sample in handlers.dataloader:
            data,label,index = sample
            blob['data'].append(data)
            blob['label'].append(label)
            blob['index'].append(index)
            minibatch_ctr += 1
                
            if minibatch_ctr % (flags.BATCH_SIZE/flags.MINIBATCH_SIZE) > 0: continue
                
            epoch = handlers.iteration * float(flags.BATCH_SIZE) / (len(handlers.dataloader.dataset))
            tstamp_iteration = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
            checkpt_step = flags.CHECKPOINT_STEP and flags.WEIGHT_PREFIX and ((handlers.iteration+1) % flags.CHECKPOINT_STEP == 0)

            # Train step
            res = handlers.trainer.forward(blob, batch_size=flags.BATCH_SIZE)
        
            # Save snapshot
            if checkpt_step:
                handlers.trainer.save_state(handlers.iteration)

            tspent_iteration = time.time() - tstart_iteration
            tsum += tspent_iteration
            log(handlers, tstamp_iteration, tspent_iteration, tsum, res, flags, epoch)

            # Increment iteration counter
            handlers.iteration += 1        
            if handlers.iteration >= flags.ITERATION:
                break
            tstart_iteration = time.time()
            blob={'data':[],'label':[],'index':[]}
    # Finalize
    if handlers.csv_logger:
        handlers.csv_logger.close()

