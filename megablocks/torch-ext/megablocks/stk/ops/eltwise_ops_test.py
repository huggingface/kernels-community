import unittest
import itertools
import torch
from absl.testing import parameterized

import stk
from stk.ops.linear_ops_test import allclose, _dense_and_sparse

_MATRIX_SIZES = (
    (128, 128, 0.0),
    (256, 256, 0.5),
    (2048, 1024, 0.8),
    (512, 128, 0.0),
    (128, 512, 0.0),
    (1024, 512, 0.0),
    (1024, 512, 0.5),
    (1024, 512, 0.75),
    (512, 1024, 0.0),
    (512, 1024, 0.5),
    (512, 1024, 0.75),
    (1024, 1024, 0.0),
    (1024, 1024, 0.5),
    (1024, 1024, 0.75),
)

_DTYPE = (
    torch.float16, torch.bfloat16
)

def _generate_testcases():
    testcases = itertools.product(_MATRIX_SIZES, _DTYPE)
    testcases = [(*size, 128, dtype) for 
        (size, dtype) in testcases]
    return testcases

_ELTWISE_OP_TESTS = _generate_testcases()

def _dense_and_sparse_like(x, std=0.1):
    dense_data = torch.randn_like(x.data, device=x.device) * std
    sparse = stk.Matrix(x.size(),
                        dense_data,
                        x.row_indices,
                        x.column_indices,
                        x.offsets)
    dense = stk.ops.to_dense(sparse)

    return (dense.requires_grad_(True),
            sparse.requires_grad_(True))

@parameterized.parameters(_ELTWISE_OP_TESTS)
class EltwiseOpsTest(parameterized.TestCase):

    def testEltwiseMul(self, m, n, sparsity, blocking, dtype):

        a_dense, a = _dense_and_sparse(m, n, sparsity, blocking, dtype)
        b_dense, b = _dense_and_sparse_like(a)

        out = stk.ops.mul(a, b)
        expected_out = torch.mul(a_dense, b_dense)

        # Compute the gradients w.r.t. the inputs.
        expected_out.sum().backward()
        stk.ops.sum(out).backward()

        # Validate the results.
        out = stk.ops.to_dense(out)
        self.assertEqual(out.dim(), 2)
        self.assertEqual(expected_out.size(), out.size())
        self.assertTrue(allclose(out, expected_out)) 

        # LHS gradient.
        grad = stk.ops.to_dense(a.grad)
        expected_grad = a_dense.grad
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size(), grad.size())
        self.assertTrue(allclose(grad, expected_grad))

        # RHS gradient.
        grad =  stk.ops.to_dense(b.grad)
        expected_grad = b_dense.grad
        self.assertEqual(grad.dim(), 2)
        self.assertEqual(expected_grad.size(), grad.size())
        self.assertTrue(allclose(grad, expected_grad))

if __name__ == '__main__':
    unittest.main()
