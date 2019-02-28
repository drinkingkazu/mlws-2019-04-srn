# mlws-2019-04-srn

Sparse ResNet for the workshop.

## Brief overview
This repo contains quickly put together scripts for training 3D sparse CNN for a ml workshop. Look at `models/SparseResNet10.py` for the model definition. `iotools/spd_*.py` are two simple dataloading scheme (event-wise from file vs. loading chunk in RAM). Training step is about 0.1 [s]/iteration on v100, ~80MB tensor size on GPU (yay sparse lib!). 

* Wanted:
  * shared data buffer filling for `SPDSimple` (in `spd_reader.py`), which is `torch.utils.data.DataSet` inherit,  so that the already-read data can stay in memory (sparsified so this is fairly small). Maybe w/ `multiprocessing` module?
  * data augmentation probably needed heavily, or generate or variety of data.
  * point regression is probably a fun extension for the workshop.
  * improve data formatting into sparse 3d tensor: grouping of timing axis not optimal, good to use an idea based on actual geometry.
  * implement graph conv as another way to tackle 3d sparse data. might work better for this classification example.
  * implement segmentation net (least important, but fun if muon+gamma or some mixture event can be generated and separate them at pixel level).
  
## Requirement
* Software ... Use [this](https://www.singularity-hub.org/containers/6596) Singularity container or [this](https://hub.docker.com/r/deeplearnphysics/larcv2) Docker container (tag `cuda90-pytorch-dev20181015-scn`).
  * Refer to [this](https://github.com/DeepLearnPhysics/larcv2-docker/blob/build/Dockerfile.larcv1.0.0rc01-cuda90-pytorchdev20181015-scn) recipe if you want to check more precisely about dependencies.
* Data ... consult with Patrick de Perio :)

## Running

Example command to run a training:
```
$HOME/sw/2019-02-mlws//bin/spd.py train -ld log -wp weights/snapshot -chks 100 -nc 2 -it 30000 -bs 32 -mbs 32 --gpus 0  -rs 1 -mn SimpleResNet10 -io spd_buffer -id /scratch/kterao/hkml_data/eminus,/scratch/kterao/hkml_data/gamma/ -nr 10 -dd 3 -ss 128 -uf 16
```

Example command to run an inference (you'll need your own weights):
```
$HOME/sw/2019-02-mlws/bin/spd.py inference -ld log -nc 2 -it 300 -bs 32 -mbs 32 --gpus 0  -rs 1 -mn SimpleResNet10 -io spd_buffer -id /scratch/kterao/hkml_data/eminus,/scratch/kterao/hkml_data/gamma/ -nr 10 -dd 3 -ss 128 -uf 16 -mp weights/snapshot-24999.ckpt
```
