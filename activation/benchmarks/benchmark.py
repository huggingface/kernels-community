from kernels.benchmarks import SiluAndMulBenchmark


class SiluWorkloads(SiluAndMulBenchmark):
    pass


class SiluWorkloadsGraph(SiluAndMulBenchmark):
    cuda_graph = True
