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
var dl = require("deeplearn");
var benchmark_util = require("./benchmark_util");
function getUnaryOp(option) {
    switch (option) {
        case 'log':
            return function (x) { return x.log(); };
        case 'exp':
            return function (x) { return x.exp(); };
        case 'neg':
            return function (x) { return x.neg(); };
        case 'ceil':
            return function (x) { return x.ceil(); };
        case 'floor':
            return function (x) { return x.floor(); };
        case 'log1p':
            return function (x) { return x.log1p(); };
        case 'sqrt':
            return function (x) { return x.sqrt(); };
        case 'square':
            return function (x) { return x.square(); };
        case 'abs':
            return function (x) { return x.abs(); };
        case 'relu':
            return function (x) { return x.relu(); };
        case 'elu':
            return function (x) { return x.elu(); };
        case 'selu':
            return function (x) { return x.selu(); };
        case 'leakyRelu':
            return function (x) { return x.leakyRelu(); };
        case 'prelu':
            // TODO: Configurable from UI
            var alpha_1 = dl.scalar(0.1);
            return function (x) { return x.prelu(alpha_1); };
        case 'sigmoid':
            return function (x) { return x.sigmoid(); };
        case 'sin':
            return function (x) { return x.sin(); };
        case 'cos':
            return function (x) { return x.cos(); };
        case 'tan':
            return function (x) { return x.tan(); };
        case 'asin':
            return function (x) { return x.asin(); };
        case 'acos':
            return function (x) { return x.acos(); };
        case 'atan':
            return function (x) { return x.atan(); };
        case 'sinh':
            return function (x) { return x.sinh(); };
        case 'cosh':
            return function (x) { return x.cosh(); };
        case 'tanh':
            return function (x) { return x.tanh(); };
        case 'step':
            return function (x) { return x.step(); };
        default:
            throw new Error("Not found such ops: " + option);
    }
}
var UnaryOpsCPUBenchmark = /** @class */ (function () {
    function UnaryOpsCPUBenchmark() {
    }
    UnaryOpsCPUBenchmark.prototype.run = function (size, option) {
        return __awaiter(this, void 0, void 0, function () {
            var input, op, start, end;
            return __generator(this, function (_a) {
                dl.setBackend('cpu');
                input = dl.randomUniform([size, size], -1, 1);
                op = getUnaryOp(option);
                start = performance.now();
                dl.tidy(function () {
                    op(input).get();
                });
                end = performance.now();
                return [2 /*return*/, end - start];
            });
        });
    };
    return UnaryOpsCPUBenchmark;
}());
exports.UnaryOpsCPUBenchmark = UnaryOpsCPUBenchmark;
var UnaryOpsGPUBenchmark = /** @class */ (function () {
    function UnaryOpsGPUBenchmark() {
    }
    UnaryOpsGPUBenchmark.prototype.run = function (size, option) {
        return __awaiter(this, void 0, void 0, function () {
            var input, op, benchmark, time;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        dl.setBackend('webgl');
                        input = dl.randomUniform([size, size], -1, 1);
                        op = getUnaryOp(option);
                        benchmark = function () { return op(input); };
                        return [4 /*yield*/, benchmark_util.warmupAndBenchmarkGPU(benchmark)];
                    case 1:
                        time = _a.sent();
                        input.dispose();
                        return [2 /*return*/, time];
                }
            });
        });
    };
    return UnaryOpsGPUBenchmark;
}());
exports.UnaryOpsGPUBenchmark = UnaryOpsGPUBenchmark;
