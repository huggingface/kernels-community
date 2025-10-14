import unittest

from absl.testing import parameterized
import stk
import torch


@parameterized.parameters(
    (8, 16, 0.0, 1),
    (8, 16, 0.5, 1),
    (8, 16, .95, 1),
    (16, 8, 0.0, 1),
    (16, 8, 0.5, 1),
    (16, 8, .95, 1),
    (8, 16, 0.0, 8),
    (8, 16, 0.5, 8),
    (8, 16, 1.0, 8),
    (16, 8, 0.0, 8),
    (16, 8, 0.5, 8),
    (16, 8, 1.0, 8),
    (128, 256, 0.5, 16),
    (256, 128, 0.75, 32),
    (512, 512, .875, 128))
class MatrixOpsTest(parameterized.TestCase):

    def testMatrixOps_FormatConversion(self, rows, cols, sparsity, blocking):
        mask = stk.random.dense_mask(rows, cols, sparsity, blocking)
        x = (torch.randn(rows, cols) * mask).type(torch.float16)

        # Convert the matrix to sparse format.
        sparse_x = stk.ops.to_sparse(x, blocking)

        # Validate the matrix.
        sparse_x.validate()

        # Validate the shape.
        self.assertEqual(sparse_x.dim(), 2)
        self.assertEqual(sparse_x.size()[0], rows)
        self.assertEqual(sparse_x.size()[1], cols)

        # Validate the sparsity.
        numblocks = rows // blocking * cols // blocking
        nnz = round(numblocks * (1 - sparsity)) * blocking ** 2
        self.assertEqual(sparse_x.nnz, nnz)

        # Convert back to dense format.
        dense_x = stk.ops.to_dense(sparse_x)

        # Validate the shape.
        self.assertEqual(dense_x.dim(), 2)
        self.assertEqual(dense_x.size()[0], rows)
        self.assertEqual(dense_x.size()[1], cols)

        # Validate the sparsity
        self.assertEqual(torch.count_nonzero(dense_x).item(), nnz)

        # Validate the output.
        self.assertTrue(torch.all(torch.eq(x, dense_x)))


if __name__ == '__main__':
    unittest.main()
