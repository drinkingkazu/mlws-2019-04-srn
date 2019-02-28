from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.utils.data import DataLoader
from spd.iotools.spd_reader import SPDSimple, SPDSparsify, SPDCollate
from spd.iotools.spd_buffer import SPDBuffer
def DataLoaderFactory(flags):

    if flags.IO_TYPE == 'spd_reader':
        d = SPDSimple(data_dirs=flags.INPUT_DIRS,transform=SPDSparsify(flags.SPATIAL_SIZE), limit_num_file=flags.LIMIT_NUM_FILE)
        return DataLoader(d, batch_size=flags.MINIBATCH_SIZE, shuffle=flags.SHUFFLE, num_workers=flags.NUM_READERS, collate_fn=SPDCollate)
    if flags.IO_TYPE == 'spd_buffer':
        d = SPDBuffer(data_dirs=flags.INPUT_DIRS,transform=SPDSparsify(flags.SPATIAL_SIZE), limit_num_file=flags.LIMIT_NUM_FILE)
        return DataLoader(d, batch_size=flags.MINIBATCH_SIZE, shuffle=flags.SHUFFLE, num_workers=flags.NUM_READERS, collate_fn=SPDCollate)
    raise NotImplementedError
