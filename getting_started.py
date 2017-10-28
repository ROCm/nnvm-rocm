from __future__ import absolute_import, print_function
import tvm
import numpy as np

n = tvm.var("n")
A = tvm.placeholder((n,), name='A')
B = tvm.placeholder((n,), name='B')
C = tvm.compute(A.shape, lambda i: A[i] + B[i], name="C")
s = tvm.create_schedule(C.op)
bx, tx = s[C].split(C.op.axis[0], factor=64)
s[C].bind(bx, tvm.thread_axis("blockIdx.x"))
s[C].bind(tx, tvm.thread_axis("threadIdx.x"))

fadd_rocm = tvm.build(s, [A, B, C], "rocm", target_host="llvm", name="myadd")
ctx = tvm.rocm(0)

n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fadd_rocm(a, b, c)
np.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())

dev_module = fadd_rocm.imported_modules[0]
print(dev_module.get_source("llvm"))
print(dev_module.get_source("asm"))
