"use strict";
/**
 * @license
 * Copyright 2017 Google Inc. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */
Object.defineProperty(exports, "__esModule", { value: true });
// tslint:disable-next-line:max-line-length
var batchnormalization3d_benchmark_1 = require("./batchnormalization3d_benchmark");
var benchmark_1 = require("./benchmark");
// tslint:disable-next-line:max-line-length
var conv_benchmarks_1 = require("./conv_benchmarks");
// tslint:disable-next-line:max-line-length
var matmul_benchmarks_1 = require("./matmul_benchmarks");
// tslint:disable-next-line:max-line-length
var pool_benchmarks_1 = require("./pool_benchmarks");
// tslint:disable-next-line:max-line-length
var reduction_ops_benchmark_1 = require("./reduction_ops_benchmark");
// tslint:disable-next-line:max-line-length
var unary_ops_benchmark_1 = require("./unary_ops_benchmark");
function getRunGroups() {
    var groups = [];
    groups.push({
        name: 'Batch Normalization 3D: input [size, size, 8]',
        min: 0,
        max: 512,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [
            new benchmark_1.BenchmarkRun('batchnorm3d_gpu', new batchnormalization3d_benchmark_1.BatchNormalization3DGPUBenchmark()),
            new benchmark_1.BenchmarkRun('batchnorm3d_cpu', new batchnormalization3d_benchmark_1.BatchNormalization3DCPUBenchmark())
        ],
        params: {}
    });
    groups.push({
        name: 'Matrix Multiplication: ' +
            'matmul([size, size], [size, size])',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [
            new benchmark_1.BenchmarkRun('mulmat_gpu', new matmul_benchmarks_1.MatmulGPUBenchmark()),
            new benchmark_1.BenchmarkRun('mulmat_cpu', new matmul_benchmarks_1.MatmulCPUBenchmark())
        ],
        params: {}
    });
    var convParams = { inDepth: 8, filterSize: 7, stride: 1, pad: 'same' };
    var regParams = Object.assign({}, convParams, { outDepth: 3 });
    var depthwiseParams = Object.assign({}, convParams, { channelMul: 1 });
    groups.push({
        name: 'Convolution ops [size, size, depth]',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        benchmarkRuns: [new benchmark_1.BenchmarkRun('conv_gpu', new conv_benchmarks_1.ConvGPUBenchmark())],
        options: ['regular', 'transposed', 'depthwise'],
        selectedOption: 'regular',
        params: {
            'regular': regParams,
            'transposed': regParams,
            'depthwise': depthwiseParams
        }
    });
    var poolParams = { depth: 8, fieldSize: 4, stride: 4 };
    groups.push({
        name: 'Pool Ops: input [size, size]',
        min: 0,
        max: 1024,
        stepSize: 64,
        stepToSizeTransformation: function (step) { return Math.max(4, step); },
        options: ['max', 'min', 'avg'],
        selectedOption: 'max',
        benchmarkRuns: [
            new benchmark_1.BenchmarkRun('pool_gpu', new pool_benchmarks_1.PoolGPUBenchmark()),
            new benchmark_1.BenchmarkRun('pool_cpu', new pool_benchmarks_1.PoolCPUBenchmark())
        ],
        params: { 'max': poolParams, 'min': poolParams, 'avg': poolParams }
    });
    groups.push({
        name: 'Unary Ops: input [size, size]',
        min: 0,
        max: 1024,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        options: [
            'log', 'exp', 'neg', 'ceil', 'floor', 'log1p', 'sqrt', 'square',
            'abs', 'relu', 'elu', 'selu', 'leakyRelu', 'prelu', 'sigmoid',
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh',
            'tanh', 'step'
        ],
        selectedOption: 'log',
        stepSize: 64,
        benchmarkRuns: [
            new benchmark_1.BenchmarkRun('unary ops CPU', new unary_ops_benchmark_1.UnaryOpsCPUBenchmark()),
            new benchmark_1.BenchmarkRun('unary ops GPU', new unary_ops_benchmark_1.UnaryOpsGPUBenchmark())
        ],
        params: {}
    });
    groups.push({
        name: 'Reduction Ops: input [size, size]',
        min: 0,
        max: 1024,
        stepToSizeTransformation: function (step) { return Math.max(1, step); },
        options: ['max', 'min', 'argMax', 'argMin', 'sum', 'logSumExp'],
        selectedOption: 'max',
        stepSize: 64,
        benchmarkRuns: [
            new benchmark_1.BenchmarkRun('reduction ops CPU', new reduction_ops_benchmark_1.ReductionOpsCPUBenchmark()),
            new benchmark_1.BenchmarkRun('reduction ops GPU', new reduction_ops_benchmark_1.ReductionOpsGPUBenchmark())
        ],
        params: {}
    });
    return groups;
}
exports.getRunGroups = getRunGroups;
