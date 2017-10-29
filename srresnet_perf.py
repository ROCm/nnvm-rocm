"""
Uses the SRResNet model in advanced_superres_onnx.py to
compare peformance of ROCm backend against OpenCL backend

Output on R9 Nano:
$ python srresnet_perf.py
diff: 0.000000
results identical
Target rocm, Elapsed 0.043458 ms
Target opencl, Elapsed 0.107162 ms
"""

import nnvm
import tvm
import onnx
import numpy as np
import nnvm.compiler
from tvm.contrib import graph_runtime

rescale_factor = 4
height = 64
width = 64
x = np.zeros((1, 3, height, width))
onnx_graph = onnx.load('data/srresnet/%dx%d.onnx' % (height, width))
targets = ["rocm", "opencl"]
num_iter = 100
outputs = []
elapsed = []

for target in targets:
    if not tvm.module.enabled(target):
        print("Skip benchmarking target %s." % target)
    sym, params = nnvm.frontend.from_onnx(onnx_graph)
    shape_dict = {'input_0': x.shape}
    graph, lib, params = nnvm.compiler.build(sym, target, shape_dict, params=params)

    ctx = tvm.context(target, 0)
    dtype = 'float32'
    m = graph_runtime.create(graph, lib, ctx)
    m.set_input('input_0', tvm.nd.array(x.astype(dtype)))
    m.set_input(**params)
    m.run()

    height_rescaled = height * rescale_factor
    width_rescaled = width * rescale_factor
    output_shape = (1, 3, height_rescaled, width_rescaled)
    tvm_output = m.get_output(0, tvm.nd.empty(output_shape, dtype)).asnumpy()
    outputs.append(tvm_output)

    ftimer = m.module.time_evaluator("run", ctx, num_iter)
    elapsed.append(ftimer().mean)

if len(outputs) == 2:
    print("diff: %f" % np.sum(np.abs(outputs[0] - outputs[1])))
    np.testing.assert_allclose(outputs[0], outputs[1])
    print("results identical")

for (target, t) in zip(targets, elapsed):
    print("Target %s, Elapsed %f ms" % (target, t))
