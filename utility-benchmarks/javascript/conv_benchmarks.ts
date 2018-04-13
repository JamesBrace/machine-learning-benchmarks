import * as dl from 'deeplearn';

import {BenchmarkTest} from './benchmark';
import * as benchmark_util from './benchmark_util';

export interface ConvParams {
  inDepth: number;
  filterSize: number;
  stride: number;
  pad: 'valid'|'same'|number;
}

export interface RegularConvParams extends ConvParams { outDepth: number; }

export interface DepthwiseConvParams extends ConvParams { channelMul: number; }

export class ConvBenchmark implements BenchmarkTest {
  async run(size: number, opType: string, params: ConvParams, useGpu = true): Promise<number> {

    if (useGpu){
      dl.setBackend('webgl');
    } else {
      dl.setBackend('cpu');
    }

    const inDepth = params.inDepth;
    const inShape: [number, number, number] = [size, size, inDepth];
    const filterSize = params.filterSize;
    const stride = params.stride;
    const pad = params.pad;

    let x: dl.Tensor3D = dl.randomUniform(inShape, -1, 1);
    let W: dl.Tensor4D;

    let benchmark: () => dl.Tensor;

    let start =  (useGpu) ? undefined : performance.now();

    if (opType === 'regular') {
      const regParams = params as RegularConvParams;
      const wShape: [number, number, number, number] =
          [filterSize, filterSize, inDepth, regParams.outDepth];
      W = dl.randomUniform(wShape, -1, 1);
      benchmark = () => x.conv2d(W, stride, pad);
    } else if (opType === 'transposed') {
      const regParams = params as RegularConvParams;
      const wShape: [number, number, number, number] =
          [filterSize, filterSize, inDepth, regParams.outDepth];
      W = dl.randomUniform(wShape, -1, 1);
      x = dl.randomUniform([size, size, regParams.outDepth], -1, 1);

      benchmark = () =>
          x.conv2dTranspose(W, [size, size, inDepth], stride, pad);
    } else if (opType === 'depthwise') {
      const depthwiseParams = params as DepthwiseConvParams;
      const wShape: [number, number, number, number] =
          [filterSize, filterSize, inDepth, depthwiseParams.channelMul];
      W = dl.randomUniform(wShape, -1, 1);

      benchmark = () => x.depthwiseConv2D(W, stride, pad);
    } else {
      throw new Error(`Unknown option ${opType}`);
    }

    let end = (useGpu) ? undefined : performance.now();

    const time = (useGpu) ? await benchmark_util.warmupAndBenchmarkGPU(benchmark): end - start;

    if(useGpu){
      x.dispose();
      W.dispose();
    }

    return time;
  }
}
