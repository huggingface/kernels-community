import unittest

from absl.testing import parameterized
from . import random
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
class RandomOpsTest(parameterized.TestCase):

    def testRandomOps_DenseMask(self, rows, cols, sparsity, blocking):
        mask = random.dense_mask(
            rows, cols, sparsity, blocking)

        # Validate the shape.
        self.assertEqual(mask.dim(), 2)
        self.assertEqual(mask.size()[0], rows)
        self.assertEqual(mask.size()[1], cols)

        # Validate the sparsity
        numblocks = rows // blocking * cols // blocking
        nnz = round(numblocks * (1 - sparsity)) * blocking ** 2
        self.assertEqual(
            torch.count_nonzero(mask).item(),
            nnz)

        # Check values are zero or one.
        self.assertTrue(
            torch.all(torch.logical_or(
                torch.eq(mask, 0),
                torch.eq(mask, 1))))

    def testRandomOps_SparseMask(self, rows, cols, sparsity, blocking):
        mask = random.mask(
            rows, cols, sparsity, blocking)

        # Validate the matrix.
        mask.validate()

        # Validate the shape.
        self.assertEqual(mask.dim(), 2)
        self.assertEqual(mask.size()[0], rows)
        self.assertEqual(mask.size()[1], cols)

        # Validate the sparsity.
        numblocks = rows // blocking * cols // blocking
        nnz = round(numblocks * (1 - sparsity)) * blocking ** 2
        self.assertEqual(mask.nnz, nnz)

        # Check values are zero or one.
        self.assertTrue(
            torch.all(torch.logical_or(
                torch.eq(mask.data, 0),
                torch.eq(mask.data, 1))))


if __name__ == '__main__':
    unittest.main()
