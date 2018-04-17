import * as dl from 'deeplearn';

import {BenchmarkTest} from './benchmark';
import * as benchmark_util from './utilities';

export interface ConvParams {
    inDepth: number;
    filterSize: number;
    stride: number;
    pad: 'valid'|'same'|number;
    outDepth?: number;
    channelMul?: number;
}

export class ConvCPU implements BenchmarkTest {
    async run(size: number, opType: string, params: ConvParams): Promise<number> {
        dl.setBackend('cpu');

        const inDepth = params.inDepth;
        const inShape: [number, number, number] = [size, size, inDepth];
        const filterSize = params.filterSize;
        const stride = params.stride;
        const pad = params.pad;

        let x: dl.Tensor3D = dl.randomUniform(inShape, -1, 1);
        let W: dl.Tensor4D;

        let start = performance.now();

        if (opType === 'regular') {
            const wShape: [number, number, number, number] = [filterSize, filterSize, inDepth, params.outDepth];
            W = dl.randomUniform(wShape, -1, 1);
            x.conv2d(W, stride, pad);
        } else if (opType === 'transposed') {
            const regParams = params;
            const wShape: [number, number, number, number] = [filterSize, filterSize, inDepth, regParams.outDepth];
            W = dl.randomUniform(wShape, -1, 1);
            x = dl.randomUniform([size, size, regParams.outDepth], -1, 1);
            x.conv2dTranspose(W, [size, size, inDepth], stride, pad);
        } else if (opType === 'depthwise') {
            const wShape: [number, number, number, number] = [filterSize, filterSize, inDepth, params.channelMul];
            W = dl.randomUniform(wShape, -1, 1);
            x.depthwiseConv2D(W, stride, pad);
        } else {
            throw new Error(`Unknown option ${opType}`);
        }

        let end = performance.now();
        return end - start;
    }
}

export class ConvGPU implements BenchmarkTest {
    async run(size: number, opType: string, params: ConvParams): Promise<number> {
        dl.setBackend('webgl');

        const inDepth = params.inDepth;
        const inShape: [number, number, number] = [size, size, inDepth];
        const filterSize = params.filterSize;
        const stride = params.stride;
        const pad = params.pad;

        let x: dl.Tensor3D = dl.randomUniform(inShape, -1, 1);
        let W: dl.Tensor4D;

        let benchmark: () => dl.Tensor;

        if (opType === 'regular') {
            const wShape: [number, number, number, number] = [filterSize, filterSize, inDepth, params.outDepth];
            W = dl.randomUniform(wShape, -1, 1);
            benchmark = () => x.conv2d(W, stride, pad);
        } else if (opType === 'transposed') {
            const regParams = params;
            const wShape: [number, number, number, number] = [filterSize, filterSize, inDepth, regParams.outDepth];
            W = dl.randomUniform(wShape, -1, 1);
            x = dl.randomUniform([size, size, regParams.outDepth], -1, 1);
            benchmark = () => x.conv2dTranspose(W, [size, size, inDepth], stride, pad);
        } else if (opType === 'depthwise') {
            const wShape: [number, number, number, number] = [filterSize, filterSize, inDepth, params.channelMul];
            W = dl.randomUniform(wShape, -1, 1);
            benchmark = () => x.depthwiseConv2D(W, stride, pad);
        } else {
            throw new Error(`Unknown option ${opType}`);
        }

        const time = await benchmark_util.warmupAndBenchmarkGPU(benchmark);

        x.dispose();
        W.dispose();

        return time;
    }
}

export async function run(size, type, params, backend){
    let benchmark: BenchmarkTest = (backend === 'gpu') ? new ConvGPU() : new ConvCPU();
    return await benchmark.run(size, type, params)
}
