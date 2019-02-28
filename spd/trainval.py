from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import time
import os
import sys
from   spd.iotools import GraphDataParallel
import spd.models as models
import numpy as np

class trainval(object):
    def __init__(self, flags):
        self._flags = flags
        self.tspent = {}
        self.tspent_sum = {}

    def backward(self):
        total_loss = 0.0
        for loss in self._loss:
            total_loss += loss.mean()
        total_loss /= len(self._loss)
        self._loss = []  # Reset loss accumulator

        self._optimizer.zero_grad()  # Reset gradients accumulation
        total_loss.backward()
        self._optimizer.step()

    def save_state(self, iteration):
        tstart = time.time()
        filename = '%s-%d.ckpt' % (self._flags.WEIGHT_PREFIX, iteration)
        torch.save({
            'global_step': iteration,
            'state_dict': self._net.state_dict(),
            'optimizer': self._optimizer.state_dict()
        }, filename)
        self.tspent['save'] = time.time() - tstart

    def train_step(self, blob, batch_size=1):
        tstart = time.time()
        self._loss = []  # Initialize loss accumulator
        res_combined = self.forward(blob,batch_size=batch_size)
        # Run backward once for all the previous forward
        self.backward()
        self.tspent['train'] = time.time() - tstart
        self.tspent_sum['train'] += self.tspent['train']
        return res_combined

    def forward(self, blob, batch_size=1):
        res = {'accuracy':[],'loss':[]}
        res_combined = {}
        miniblob = {'data':[],'label':[],'index':[]}
        num_proc = max(1,len(self._flags.GPUS))
        for idx in range(int(self._flags.BATCH_SIZE/self._flags.MINIBATCH_SIZE)):
            miniblob['data'].append(blob['data'][idx])
            miniblob['label'].append(blob['label'][idx])
            miniblob['index'].append(blob['index'][idx])

            if (idx+1) % num_proc > 0: continue

            res = self._forward(miniblob)
            for key in res.keys():
                if key not in res_combined:
                    res_combined[key] = res[key]
                else:
                    res_combined[key].extend(res[key])            
            miniblob = {'data':[],'label':[],'index':[]}
            
        # Average loss and acc over all the events in this batch
        res_combined['accuracy'] = np.array(res_combined['accuracy']).mean()
        res_combined['loss'    ] = np.array(res_combined['loss'    ]).mean()
        return res_combined

    def _forward(self, blob):
        """
        """
        data  = blob['data']
        label = blob.get('label',None)
        with torch.set_grad_enabled(self._flags.TRAIN):
            # Prediction
            if torch.cuda.is_available():
                data = [torch.as_tensor(d).cuda() for d in data]
            else:
                assert len(data) == 1
                data = torch.as_tensor(data[0])
            tstart = time.time()
            prediction = self._net(data)
            if isinstance(prediction,list): prediction = torch.stack(prediction)
            # Training
            loss,acc=-1,-1
            if label is not None:
                label = torch.stack([ torch.as_tensor(l) for l in np.hstack(label) ])
                if torch.cuda.is_available(): label=label.cuda()
                label.requires_grad = False
                loss = self._criterion(prediction,label)
                if self._flags.TRAIN:
                    self._loss.append(loss)
            predicted_labels = torch.argmax(prediction,dim=1)
            res = {'prediction' : [ np.argmax(pred.cpu().detach().numpy(),axis=-1) for pred in prediction],
                   'softmax'    : [ self._softmax(pred).cpu().detach().numpy() for pred in prediction],
                   'accuracy'   : [ (predicted_labels == label).cpu().detach().numpy() ],
                   'loss'       : [ loss.mean().cpu().detach().numpy() if not isinstance(loss, float) else loss ]}
            self.tspent['forward'] = time.time() - tstart
            self.tspent_sum['forward'] += self.tspent['forward']
            return res

    def initialize(self):
        # To use DataParallel all the inputs must be on devices[0] first
        model = getattr(models,self._flags.MODEL_NAME)
        self._criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.tspent_sum['forward'] = self.tspent_sum['train'] = self.tspent_sum['save'] = 0.
        self.tspent['forward'] = self.tspent['train'] = self.tspent['save'] = 0.

        # if len(self._flags.GPUS) > 0:
        self._net = GraphDataParallel(model(self._flags),device_ids=self._flags.GPUS)

        if self._flags.TRAIN:
            self._net.train()
        else:
            self._net.eval()

        if torch.cuda.is_available():
            self._net.cuda()
            self._criterion.cuda()

        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._flags.LEARNING_RATE)
        self._softmax = torch.nn.Softmax(dim=0)

        iteration = 0
        if self._flags.MODEL_PATH:
            if not os.path.isfile(self._flags.MODEL_PATH):
                sys.stderr.write('File not found: %s\n' % self._flags.MODEL_PATH)
                raise ValueError
            print('Restoring weights from %s...' % self._flags.MODEL_PATH)
            with open(self._flags.MODEL_PATH, 'rb') as f:
                checkpoint = torch.load(f)
                self._net.load_state_dict(checkpoint['state_dict'], strict=False)
                if self._flags.TRAIN:
                    # This overwrites the learning rate, so reset the learning rate
                    self._optimizer.load_state_dict(checkpoint['optimizer'])
                    for g in self._optimizer.param_groups:
                        g['lr'] = self._flags.LEARNING_RATE
                iteration = checkpoint['global_step'] + 1
            print('Done.')

        return iteration
