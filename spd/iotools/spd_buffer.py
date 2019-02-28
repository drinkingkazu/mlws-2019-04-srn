from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader

class SPDBuffer(Dataset):

    def __init__(self, data_dirs, transform=None, limit_num_file=0):

        self._data  = []
        self._label = []

        kilo_ctr=0
        for d in data_dirs:
            flist = None
            if limit_num_file > 0:
                flist = [ os.path.join(d,f) for f in os.listdir(d)[0:limit_num_file] ]
            else:
                flist = [ os.path.join(d,f) for f in os.listdir(d) ]
            for f in flist:
                df = np.load(f)
                label_v = df['labels']
                data_v  = df['event_data']
                # hack
                label_v = [int('gamma' in f) for _ in range(len(label_v))]
                for idx in range(len(label_v)):
                    label = label_v[idx]
                    data  = data_v[idx]
                    if transform is not None:
                        data = transform(data)
                    if data.shape[0] < 10: continue
                    self._label.append(label)
                    self._data.append(data)
                    
                if int(len(self._label)/1000.) > kilo_ctr:
                    kilo_ctr = int(len(self._label)/1000.)
                    print('Processed',kilo_ctr)

        val,ctr=np.unique(self._label,return_counts=True)
        print('Label values:',val)
        print('Statistics:',ctr)
    def __len__(self):
        return len(self._data)
            
    def __getitem__(self,idx):
        return self._data[idx],self._label[idx],idx

