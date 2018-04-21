"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = y[op[0] & 2 ? "return" : op[0] ? "throw" : "next"]) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [0, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
Object.defineProperty(exports, "__esModule", { value: true });
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
var dl = require("deeplearn");
var benchmark_1 = require("./benchmark");
var benchmark_util = require("./utilities");
var BatchNormalization3DCPUBenchmark = /** @class */ (function () {
    function BatchNormalization3DCPUBenchmark() {
    }
    BatchNormalization3DCPUBenchmark.prototype.run = function (size) {
        return __awaiter(this, void 0, void 0, function () {
            var x, mean, variance, varianceEpsilon, start, end;
            return __generator(this, function (_a) {
                if (this.lastRunTimeMs > benchmark_1.LAST_RUN_CPU_CUTOFF_MS) {
                    return [2 /*return*/, new Promise(function (resolve, reject) {
                            resolve(-1);
                        })];
                }
                dl.setBackend('cpu');
                x = dl.randomUniform([size, size, 8], -1, 1);
                mean = dl.tensor1d([0]);
                variance = dl.tensor1d([1]);
                varianceEpsilon = .001;
                start = performance.now();
                x.batchNormalization(mean, variance, varianceEpsilon);
                end = performance.now();
                this.lastRunTimeMs = end - start;
                return [2 /*return*/, this.lastRunTimeMs];
            });
        });
    };
    return BatchNormalization3DCPUBenchmark;
}());
exports.BatchNormalization3DCPUBenchmark = BatchNormalization3DCPUBenchmark;
var BatchNormalization3DGPUBenchmark = /** @class */ (function () {
    function BatchNormalization3DGPUBenchmark() {
    }
    BatchNormalization3DGPUBenchmark.prototype.run = function (size) {
        return __awaiter(this, void 0, void 0, function () {
            var x, mean, variance, varianceEpsilon, benchmark, time;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        dl.setBackend('webgl');
                        x = dl.randomUniform([size, size, 8], -1, 1);
                        mean = dl.tensor1d([0]);
                        variance = dl.tensor1d([1]);
                        varianceEpsilon = .001;
                        benchmark = function () {
                            return x.batchNormalization(mean, variance, varianceEpsilon);
                        };
                        return [4 /*yield*/, benchmark_util.warmupAndBenchmarkGPU(benchmark)];
                    case 1:
                        time = _a.sent();
                        x.dispose();
                        mean.dispose();
                        variance.dispose();
                        return [2 /*return*/, time];
                }
            });
        });
    };
    return BatchNormalization3DGPUBenchmark;
}());
exports.BatchNormalization3DGPUBenchmark = BatchNormalization3DGPUBenchmark;
