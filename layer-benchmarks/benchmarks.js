"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
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
require("../demo-header");
require("../demo-footer");
var polymer_spec_1 = require("../polymer-spec");
var math_benchmark_run_groups_1 = require("./math-benchmark-run-groups");
// tslint:disable-next-line:variable-name
exports.MathBenchmarkPolymer = polymer_spec_1.PolymerElement({ is: 'math-benchmark', properties: { benchmarks: Array } });
var MathBenchmark = /** @class */ (function (_super) {
    __extends(MathBenchmark, _super);
    function MathBenchmark() {
        return _super !== null && _super.apply(this, arguments) || this;
    }
    MathBenchmark.prototype.ready = function () {
        var _this = this;
        var groups = math_benchmark_run_groups_1.getRunGroups();
        // Set up the benchmarks UI.
        var benchmarks = [];
        this.stopMessages = [];
        groups.forEach(function (group) {
            if (group.selectedOption == null) {
                group.selectedOption = '';
            }
            benchmarks.push(group);
            _this.stopMessages.push(false);
        });
        this.benchmarks = benchmarks;
        // In a setTimeout to let the UI update before we add event listeners.
        setTimeout(function () {
            var runButtons = _this.querySelectorAll('.run-test');
            var stopButtons = _this.querySelectorAll('.run-stop');
            var _loop_1 = function (i) {
                runButtons[i].addEventListener('click', function () {
                    _this.runBenchmarkGroup(groups, i);
                });
                stopButtons[i].addEventListener('click', function () {
                    _this.stopMessages[i] = true;
                });
            };
            for (var i = 0; i < runButtons.length; i++) {
                _loop_1(i);
            }
        }, 0);
    };
    MathBenchmark.prototype.getDisplayParams = function (paramsMap, selectedOption) {
        var params = paramsMap[selectedOption];
        if (params == null) {
            return '';
        }
        var kvParams = params;
        var out = [];
        var keys = Object.keys(kvParams);
        if (keys.length === 0) {
            return '';
        }
        for (var i = 0; i < keys.length; i++) {
            out.push(keys[i] + ': ' + kvParams[keys[i]]);
        }
        return '{' + out.join(', ') + '}';
    };
    MathBenchmark.prototype.runBenchmarkGroup = function (groups, benchmarkRunGroupIndex) {
        var benchmarkRunGroup = groups[benchmarkRunGroupIndex];
        var canvas = this.querySelectorAll('.run-plot')[benchmarkRunGroupIndex];
        // Avoid to growing size of rendered chart.
        canvas.width = 360;
        canvas.height = 270;
        var context = canvas.getContext('2d');
        var datasets = [];
        for (var i = 0; i < benchmarkRunGroup.benchmarkRuns.length; i++) {
            benchmarkRunGroup.benchmarkRuns[i].clearChartData();
            var hue = Math.floor(360 * i / benchmarkRunGroup.benchmarkRuns.length);
            datasets.push({
                data: benchmarkRunGroup.benchmarkRuns[i].chartData,
                fill: false,
                label: benchmarkRunGroup.benchmarkRuns[i].name,
                borderColor: "hsl(" + hue + ", 100%, 40%)",
                backgroundColor: "hsl(" + hue + ", 100%, 70%)",
                pointRadius: 0,
                pointHitRadius: 5,
                borderWidth: 1,
                lineTension: 0
            });
        }
        var chart = new Chart(context, {
            type: 'line',
            data: { datasets: datasets },
            options: {
                animation: { duration: 0 },
                responsive: false,
                scales: {
                    xAxes: [{
                            type: 'linear',
                            position: 'bottom',
                            ticks: {
                                min: benchmarkRunGroup.min,
                                max: benchmarkRunGroup.max,
                                stepSize: benchmarkRunGroup.stepSize,
                                callback: function (label) {
                                    return benchmarkRunGroup.stepToSizeTransformation != null ?
                                        benchmarkRunGroup.stepToSizeTransformation(+label) :
                                        +label;
                                }
                                // tslint:disable-next-line:no-any
                            } // Note: the typings for this are incorrect, cast as any.
                        }],
                    yAxes: [{
                            ticks: {
                                callback: function (label, index, labels) {
                                    return label + "ms";
                                }
                            },
                        }]
                },
                tooltips: { mode: 'label' },
                title: { text: benchmarkRunGroup.name }
            }
        });
        canvas.style.display = 'none';
        var runMessage = this.querySelectorAll('.run-message')[benchmarkRunGroupIndex];
        runMessage.style.display = 'block';
        var runNumbersTable = this.querySelectorAll('.run-numbers-table')[benchmarkRunGroupIndex];
        runNumbersTable.innerHTML = '';
        runNumbersTable.style.display = 'none';
        // Set up the header for the table.
        var headers = ['size'];
        for (var i = 0; i < benchmarkRunGroup.benchmarkRuns.length; i++) {
            headers.push(benchmarkRunGroup.benchmarkRuns[i].name);
        }
        runNumbersTable.appendChild(this.buildRunNumbersRow(headers));
        this.runBenchmarkSteps(chart, benchmarkRunGroup, benchmarkRunGroupIndex, benchmarkRunGroup.min);
    };
    MathBenchmark.prototype.buildRunNumbersRow = function (values) {
        var runNumberRowElement = document.createElement('div');
        runNumberRowElement.className = 'run-numbers-row math-benchmark';
        for (var i = 0; i < values.length; i++) {
            var runNumberCellElement = document.createElement('div');
            runNumberCellElement.className = 'run-numbers-cell math-benchmark';
            runNumberCellElement.innerText = values[i];
            runNumberRowElement.appendChild(runNumberCellElement);
        }
        return runNumberRowElement;
    };
    MathBenchmark.prototype.runBenchmarkSteps = function (chart, runGroup, benchmarkRunGroupIndex, step) {
        return __awaiter(this, void 0, void 0, function () {
            var _this = this;
            var runNumbersTable, canvas, runMessage, runNumberRowElement, rowValues, i, run, test, size, opType, time, resultString;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        runNumbersTable = this.querySelectorAll('.run-numbers-table')[benchmarkRunGroupIndex];
                        if (step > runGroup.max || this.stopMessages[benchmarkRunGroupIndex]) {
                            this.stopMessages[benchmarkRunGroupIndex] = false;
                            runNumbersTable.style.display = '';
                            canvas = this.querySelectorAll('.run-plot')[benchmarkRunGroupIndex];
                            canvas.style.display = 'block';
                            chart.update();
                            runMessage = this.querySelectorAll('.run-message')[benchmarkRunGroupIndex];
                            runMessage.style.display = 'none';
                            return [2 /*return*/];
                        }
                        runNumberRowElement = document.createElement('div');
                        runNumberRowElement.className = 'run-numbers-row math-benchmark';
                        rowValues = [step.toString()];
                        i = 0;
                        _a.label = 1;
                    case 1:
                        if (!(i < runGroup.benchmarkRuns.length)) return [3 /*break*/, 4];
                        run = runGroup.benchmarkRuns[i];
                        test = run.benchmarkTest;
                        size = runGroup.stepToSizeTransformation != null ?
                            runGroup.stepToSizeTransformation(step) :
                            step;
                        opType = runGroup.selectedOption;
                        return [4 /*yield*/, test.run(size, opType, runGroup.params[opType])];
                    case 2:
                        time = _a.sent();
                        resultString = time.toFixed(3) + 'ms';
                        if (time >= 0) {
                            run.chartData.push({ x: step, y: time });
                            rowValues.push(resultString);
                        }
                        console.log(run.name + "[" + size + "]: " + resultString);
                        _a.label = 3;
                    case 3:
                        i++;
                        return [3 /*break*/, 1];
                    case 4:
                        runNumbersTable.appendChild(this.buildRunNumbersRow(rowValues));
                        step += runGroup.stepSize;
                        // Allow the UI to update.
                        setTimeout(function () { return _this.runBenchmarkSteps(chart, runGroup, benchmarkRunGroupIndex, step); }, 100);
                        return [2 /*return*/];
                }
            });
        });
    };
    return MathBenchmark;
}(exports.MathBenchmarkPolymer));
exports.MathBenchmark = MathBenchmark;
document.registerElement(MathBenchmark.prototype.is, MathBenchmark);
