"""
Compile MXNet Models
====================
**Author**: `Joshua Z. Zhang <https://zhreshold.github.io/>`_

This article is an introductory tutorial to deploy mxnet models with NNVM.

For us to begin with, mxnet module is required to be installed.

A quick solution is
```
pip install mxnet --user
```
or please refer to offical installation guide.
https://mxnet.incubator.apache.org/versions/master/install/index.html
"""
# some standard imports
import mxnet as mx
import nnvm
import tvm
import numpy as np

######################################################################
# Download Resnet18 model from Gluon Model Zoo
# ---------------------------------------------
# In this section, we download a pretrained imagenet model and classify an image.
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.gluon.utils import download
from PIL import Image
from matplotlib import pyplot as plt

for model in ["resnet18_v1", "alexnet", "resnet50_v1"]:
    print("Testing model %s" % model)
    block = get_model(model, pretrained=True)
    img_name = 'cat.jpg'
    synset_url = ''.join(['https://gist.githubusercontent.com/zhreshold/',
                          '4d0b62f3d01426887599d4f7ede23ee5/raw/',
                          '596b27d23537e5a1b5751d2b0481ef172f58b539/',
                          'imagenet1000_clsid_to_human.txt'])
    synset_name = 'synset.txt'
    download('https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true', img_name)
    download(synset_url, synset_name)
    with open(synset_name) as f:
        synset = eval(f.read())
    image = Image.open(img_name).resize((224, 224))
    # plt.imshow(image)
    # plt.show()

    def transform_image(image):
        image = np.array(image) - np.array([123., 117., 104.])
        image /= np.array([58.395, 57.12, 57.375])
        image = image.transpose((2, 0, 1))
        image = image[np.newaxis, :]
        return image

    x = transform_image(image)
    print('x', x.shape)

    ######################################################################
    # Compile the Graph
    # -----------------
    # Now we would like to port the Gluon model to a portable computational graph.
    # It's as easy as several lines.
    # We support MXNet static graph(symbol) and HybridBlock in mxnet.gluon
    sym, params = nnvm.frontend.from_mxnet(block)
    # we want a probability so add a softmax operator
    sym = nnvm.sym.softmax(sym)

    ######################################################################
    # now compile the graph
    import nnvm.compiler
    target = 'rocm'
    shape_dict = {'data': x.shape}
    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

    ######################################################################
    # Execute the portable graph on TVM
    # ---------------------------------
    # Now, we would like to reproduce the same forward computation using TVM.
    from tvm.contrib import graph_runtime
    ctx = tvm.rocm(0)
    dtype = 'float32'
    m = graph_runtime.create(graph, lib, ctx)
    # set inputs
    m.set_input('data', tvm.nd.array(x.astype(dtype)))
    m.set_input(**params)
    # execute
    m.run()
    # get outputs
    tvm_output = m.get_output(0, tvm.nd.empty((1000,), dtype))
    top1 = np.argmax(tvm_output.asnumpy())
    print('TVM prediction top-1:', top1, synset[top1])
