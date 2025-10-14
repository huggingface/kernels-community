from ..matrix import Matrix

def mul(a, b):
    """Performs element-wise multiplication of matrices a and b.

    It is the user's responsibility to make sure that a and b
    follow the same matrix topology. This function assumes it is safe
    to use the topoplogy of a.

    Args:
        a: stk.Matrix.
        b: stk.Matrix with a's matrix topology.

    Returns:
        stk.Matrix where the entries correspond to torch.mul(a, b).
    """
    assert isinstance(a, Matrix)
    assert isinstance(b, Matrix)
    assert a.size() == b.size()

    return Matrix(a.size(),
                  a.data * b.data,
                  a.row_indices,
                  a.column_indices,
                  a.offsets,
                  a.column_indices_t,
                  a.offsets_t,
                  a.block_offsets_t)
