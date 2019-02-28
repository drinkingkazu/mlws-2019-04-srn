from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from multiprocessing import Process, Array, Value


class SPDSimple(Dataset):

    def __init__(self, data_dirs, transform=None, limit_num_file=0):
        self._transform = transform
        self._files = []
        if limit_num_file > 0:
            for d in data_dirs: self._files += [ os.path.join(d,f) for f in os.listdir(d)[0:limit_num_file] ]
        else:
            for d in data_dirs: self._files += [ os.path.join(d,f) for f in os.listdir(d) ]
        num_events_v = [np.load(f)['labels'].shape[0] for f in self._files]
        length = np.sum(num_events_v)
        self._file_index  = np.zeros([length],dtype=np.int32)
        self._event_index = np.zeros([length],dtype=np.int32)
        ctr=0
        for findex,num_events in enumerate(num_events_v):
            self._file_index  [ctr:ctr+num_events] = findex
            self._event_index [ctr:ctr+num_events] = np.arange(num_events)
            ctr += num_events

    def __len__(self):
        return len(self._file_index)
            
    def __getitem__(self,idx):

        t0=time.time()
        f = self._files[self._file_index[idx]]
        label = int('gamma' in f)
        f = np.load(f)
        i = self._event_index[idx]

        #label = f['labels'][i]
        data  = f['event_data'][i]
        if self._transform is not None:
            data = self._transform(data)

        return data,label,idx

class SPDSparsify(object):

    def __init__(self, duration):
        assert isinstance(duration, int)
        assert duration > 0
        self._duration = int(duration)
    
    def __call__(self,data):
        qs = data[:,:,0:19]
        ts = data[:,:,19:38]
        # find first hit
        t0 = np.min(ts[np.where(ts>0)])
        trange = np.where( abs(ts - t0 - self._duration/2.) <= self._duration/2. )
        data = np.zeros([np.shape(trange)[1],5],dtype=np.float32)
        data[:,0] = trange[:][0]
        data[:,1] = trange[:][1]
        data[:,2] = np.floor(ts[trange] - t0)
        data[:,4] = qs[trange]
        return data

def SPDCollate(batch):
    t=time.time()
    for index,data in enumerate(batch): data[0][:,3] = float(index)
    data  = np.vstack([sample[0] for sample in batch])
    label = [sample[1] for sample in batch]
    index = [sample[2] for sample in batch]
    t=time.time()-t
    print(t)
    return data,label,index

if __name__ == '__main__':

    import time
    t  = time.time()
    d0 = SPDSimple(data_dirs=['/scratch/kterao/hkml_data/gamma/','/scratch/kterao/hkml_data/eminus'])
    t  = time.time() - t
    print('Instantiation:', t,'[s]')

    t  = time.time()
    data,label,idx = d0[0]
    t  = time.time() - t
    print('Retrieval:',t,'[s]')


    import time
    t  = time.time()
    d1 = SPDSimple(data_dirs=['/scratch/kterao/hkml_data/gamma/','/scratch/kterao/hkml_data/eminus'],transform=SPDSparsify(100))
    t  = time.time() - t
    print('Instantiation:', t,'[s]')

    t  = time.time()
    data,label,idx = d1[0]
    t  = time.time() - t
    print('Retrieval:',t,'[s]')

    BATCH_SIZE=5
    NUM_WORKERS=10
    print(len(d1))
    dataloader = DataLoader(d1, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=SPDCollate)
    print(len(dataloader))
    t0=time.time()
    for i, sample in enumerate(dataloader):
        data,label,index = sample
        t1=time.time()
        print('Batch',i,'time',t1-t0,'[s]')
        assert len(label) == BATCH_SIZE
        
        if i==100: break
        t0=time.time()
    time.sleep(1)
