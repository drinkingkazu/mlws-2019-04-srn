from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import time

def SparseResNet(dimension, nInputPlanes, layers):
    import sparseconvnet as scn
    """
    pre-activated ResNet
    e.g. layers = {{'basic',16,2,1},{'basic',32,2}}
    """
    nPlanes = nInputPlanes
    m = scn.Sequential()
    
    def residual(nIn, nOut, stride):
        if stride > 1:
            return scn.Convolution(dimension, nIn, nOut, 2, stride, False)
        elif nIn != nOut:
            return scn.NetworkInNetwork(nIn, nOut, False)
        else:
            return scn.Identity()
    for n, reps, stride in layers:
        for rep in range(reps):
            if rep == 0:
                m.add(scn.BatchNormReLU(nPlanes))
                tab = scn.ConcatTable()
                tab_seq = scn.Sequential()
                if stride == 1:
                    tab_seq.add(scn.SubmanifoldConvolution(dimension,nPlanes,n,3,False))
                else:
                    tab_seq.add(scn.Convolution(dimension,nPlanes,n,2,stride,False))
                tab_seq.add(scn.BatchNormReLU(n))
                tab_seq.add(scn.SubmanifoldConvolution(dimension,n,n,3,False))
                tab.add(tab_seq)
                tab.add(residual(nPlanes,n,stride))                    
                m.add(tab)
            else:
                tab=scn.ConcatTable()
                tab_seq=scn.Sequential()
                tab_seq.add(scn.BatchNormReLU(nPlanes))
                tab_seq.add(scn.SubmanifoldConvolution(dimension,nPlanes,n,3,False))
                tab_seq.add(scn.BatchNormReLU(n))
                tab_seq.add(scn.SubmanifoldConvolution(dimension,n,n,3,False))
                tab.add(tab_seq)
                tab.add(scn.Identity())
                m.add(tab)
            nPlanes = n
            m.add(scn.AddTable())
    m.add(scn.BatchNormReLU(nPlanes))
    return m
                    
class SimpleResNet10(torch.nn.Module):
    
    def __init__(self,flags):
        torch.nn.Module.__init__(self)
        import sparseconvnet as scn
        self._flags = flags
        dimension  = self._flags.DATA_DIM
        num_class  = self._flags.NUM_CLASS
        image_size = self._flags.SPATIAL_SIZE
        num_filter = self._flags.BASE_NUM_FILTERS
        assert image_size == 128
        net = scn.Sequential()
        net.add(scn.InputLayer(dimension,image_size,mode=3))
        net.add(scn.SubmanifoldConvolution(dimension,1,num_filter,3,False))
        net.add(scn.MaxPooling(dimension,2,2))
        net.add(SparseResNet(dimension, num_filter, [[num_filter*1,2,1],
                                                     [num_filter*2,2,2],
                                                     [num_filter*4,2,2],
                                                     [num_filter*8,2,2]]))
        net.add(scn.Convolution(dimension, num_filter*8, num_filter*16, 3, 1, False))
        net.add(scn.BatchNormReLU(num_filter*16))
        net.add(scn.SparseToDense(dimension,num_filter*16))
        net.add(torch.nn.AvgPool3d(6))
        self._net   = net
        self.linear = torch.nn.Linear(num_filter*16,num_class)
        
    def forward(self, blob):
        voxels = blob[:,0:4]
        features = blob[:,4:]
        tensor = self._net((voxels,features))
        tensor = tensor.view(-1,self._flags.BASE_NUM_FILTERS*16)
        tensor = self.linear(tensor)
        return tensor

if __name__ == '__main__':
    class FLAGS:
        pass
    flags = FLAGS()
    flags.DATA_DIM=3
    flags.NUM_CLASS=2
    flags.SPATIAL_SIZE=128
    flags.BASE_NUM_FILTERS=16
    
    import numpy as np
    voxels   = np.zeros(shape=[200,4],dtype=np.float32)
    features = np.zeros(shape=[200,1],dtype=np.float32)
    voxels[0:100][3] = 1
    voxels[150:200][3] = 2
    model = SimpleResNet10(flags)
    data  = model.forward(torch.as_tensor(voxels),torch.as_tensor(features))
    print(data.shape)
