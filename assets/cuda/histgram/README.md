CUDA with Torch 初步尝试
---
The nn_pillar retrieves the nearest neighbor of a given point in src from dst within in a particualr pillar. This implementation takes the chamfer3D implementation as the template. Changes have been made wherever necessary.

## Install
```bash
# change it if you use different cuda version
export PATH=/usr/local/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64:$LD_LIBRARY_PATH

# Install
cd assets/cuda/nn_pillar
python ./setup.py install

```