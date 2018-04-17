import * as dl from 'deeplearn';

import {BenchmarkTest} from './benchmark';
import * as benchmark_util from './utilities';

function getReductionOp(option: string): (x: dl.Tensor) => dl.Scalar {
    switch (option) {
        case 'argMax':
            return x => x.argMax();
        case 'argMin':
            return x => x.argMin();
        case 'sum':
            return x => x.sum();
        case 'logSumExp':
            return x => x.logSumExp();
        case 'mean':
            return x => x.mean();
        default:
            throw new Error(`Not found such ops: ${option}`);
    }
}

export class ReductionOpsCPU implements BenchmarkTest {
    async run(size: number, option: string): Promise<number> {
        dl.setBackend('cpu');

        const input: dl.Tensor2D = dl.randomUniform([size, size], -1, 1);
        const op = getReductionOp(option);
        const start = performance.now();

        op(input).get();

        const end = performance.now();
        return end - start;
    }
}

export class ReductionOpsGPU implements BenchmarkTest {
    async run(size: number, option: string) {
        dl.setBackend('webgl');

        const input: dl.Tensor2D = dl.randomUniform([size, size], -1, 1);
        const op = getReductionOp(option);

        const benchmark = () => op(input);

        const time = await benchmark_util.warmupAndBenchmarkGPU(benchmark);

        input.dispose();

        return time;
    }
}

export async function run(size, option, backend){
    let benchmark: BenchmarkTest = (backend === 'gpu') ? new ReductionOpsGPU() : new ReductionOpsCPU();
    return await benchmark.run(size, option)
}
