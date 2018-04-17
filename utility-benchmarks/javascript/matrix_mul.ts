import * as dl from 'deeplearn';

import {BenchmarkTest} from './benchmark';
import * as benchmark_util from './utilities';

export class MatrixMulCPU implements BenchmarkTest {
    async run(size: number): Promise<number> {
        dl.setBackend('cpu');
        const a: dl.Tensor2D = dl.randomUniform([size, size], -1, 1);
        const b: dl.Tensor2D = dl.randomUniform([size, size], -1, 1);
        const start = performance.now();
        dl.matMul(a, b);
        const end = performance.now();
        return end - start;
    }
}

export class MatrixMulGPU implements BenchmarkTest {
    async run(size: number): Promise<number> {
        dl.setBackend('webgl');

        const a: dl.Tensor2D = dl.randomNormal([size, size]);
        const b: dl.Tensor2D = dl.randomNormal([size, size]);

        const benchmark = () => dl.matMul(a, b);

        const time = await benchmark_util.warmupAndBenchmarkGPU(benchmark);

        a.dispose();
        b.dispose();

        return time;
    }
}

export async function run(size, type, params, backend){
    let benchmark: BenchmarkTest = (backend === 'gpu') ? new MatrixMulGPU() : new MatrixMulCPU();
    return await benchmark.run(size)
}
