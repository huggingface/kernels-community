from kernels.benchmarks import (
    FlashAttentionBenchmark,
    FlashAttentionCausalBenchmark,
    FlashAttentionVarlenBenchmark,
)


class FlashAttn(FlashAttentionBenchmark):
    pass


class FlashAttnCausal(FlashAttentionCausalBenchmark):
    pass


class FlashAttnVarlen(FlashAttentionVarlenBenchmark):
    pass
