import * as dl from 'deeplearn';

import {BenchmarkTest} from './benchmark';
import * as benchmark_util from './utilities';

export class BatchNormalizationCPU implements BenchmarkTest {
    async run(size: number): Promise<number> {
        dl.setBackend('cpu');
        const x: dl.Tensor3D = dl.randomUniform([size, size, 8], -1, 1);
        const mean = dl.tensor1d([0]);
        const variance = dl.tensor1d([1]);
        const varianceEpsilon = .001;
        const start = performance.now();

        x.batchNormalization(mean, variance, varianceEpsilon);

        const end = performance.now();
        return end - start;
    }
}

export class BatchNormalizationGPU implements BenchmarkTest {
    async run(size: number) {
        dl.setBackend('webgl');

        const x: dl.Tensor3D = dl.randomUniform([size, size, 8], -1, 1);
        const mean = dl.tensor1d([0]);
        const variance = dl.tensor1d([1]);
        const varianceEpsilon = .001;

        const benchmark = () => x.batchNormalization(mean, variance, varianceEpsilon);

        const time = await benchmark_util.warmupAndBenchmarkGPU(benchmark);

        x.dispose();
        mean.dispose();
        variance.dispose();

        return time;
    }
}


export async function run(size, backend){
    let benchmark: BenchmarkTest = (backend === 'gpu') ? new BatchNormalizationGPU() : new BatchNormalizationCPU();
    return await benchmark.run(size)
}
