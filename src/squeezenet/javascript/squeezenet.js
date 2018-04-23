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
// // tslint:disable-next-line:max-line-length
var dl = require("deeplearn");
/*****************np
 * CONSTANTS
 ****************/
var log = console.log;
// Hyper-parameters
var LEARNING_RATE = .001;
var BATCH_SIZE = 64;
var TRAIN_STEPS = 1000;
// Data constants.
var optimizer = dl.train.adam(LEARNING_RATE);
// const TRAINING_SIZE = 8000;
// const TEST_SIZE = 2000;
var IMAGE_SIZE = 32;
var CIFAR10 = window.CIFAR10 || {};
var data = { training: { images: [], labels: [], num_images: 0 }, test: { images: [], labels: [], num_images: 0 } };
function loadData() {
    return __awaiter(this, void 0, void 0, function () {
        var training, test;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4 /*yield*/, CIFAR10.training.get(70)];
                case 1:
                    training = _a.sent();
                    return [4 /*yield*/, CIFAR10.test.get(70)];
                case 2:
                    test = _a.sent();
                    log(training);
                    log(test);
                    data = {
                        training: {
                            images: training.map(function (obj) { return obj.input; }),
                            labels: training.map(function (obj) { return obj.output; }),
                            num_images: training.length,
                        },
                        test: {
                            images: test.map(function (obj) { return obj.input; }),
                            labels: test.map(function (obj) { return obj.output; }),
                            num_images: test.length,
                        }
                    };
                    log(data.training.images[0]);
                    log(data.training.labels[0]);
                    return [2 /*return*/];
            }
        });
    });
}
/*****************
 * WEIGHTS
 ****************/
// Conv 1 weights
var conv1Weights = dl.variable(dl.randomNormal([2, 2, 3, 96], 0, 0.1));
var conv1Bias = dl.variable(dl.zeros([64, 32, 32, 96]));
// Fire 1 weights
var fire2SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 96, 16], 0, 0.1));
var fire2SqueezeBias = dl.variable(dl.zeros([64, 16, 16, 16]));
var fire2Expand1Weights = dl.variable(dl.randomNormal([1, 1, 16, 64], 0, 0.1));
var fire2Expand1Bias = dl.variable(dl.zeros([64, 16, 16, 64]));
var fire2Expand2Weights = dl.variable(dl.randomNormal([3, 3, 16, 64], 0, 0.1));
var fire2Expand2Bias = dl.variable(dl.randomNormal([64, 16, 16, 64], 0, 0.1));
// Fire 2 weights
var fire3SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 128, 16], 0, 0.1));
var fire3SqueezeBias = dl.variable(dl.zeros([64, 16, 16, 16]));
var fire3Expand1Weights = dl.variable(dl.randomNormal([1, 1, 16, 64], 0, 0.1));
var fire3Expand1Bias = dl.variable(dl.zeros([64, 16, 16, 64]));
var fire3Expand2Weights = dl.variable(dl.randomNormal([3, 3, 16, 64], 0, 0.1));
var fire3Expand2Bias = dl.variable(dl.zeros([64, 16, 16, 64]));
// Fire 3 weights
var fire4SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 128, 32], 0, 0.1));
var fire4SqueezeBias = dl.variable(dl.zeros([64, 16, 16, 32]));
var fire4Expand1Weights = dl.variable(dl.randomNormal([1, 1, 32, 128], 0, 0.1));
var fire4Expand1Bias = dl.variable(dl.zeros([64, 16, 16, 128]));
var fire4Expand2Weights = dl.variable(dl.randomNormal([3, 3, 32, 128], 0, 0.1));
var fire4Expand2Bias = dl.variable(dl.zeros([64, 16, 16, 128]));
// Fire 4 weights
var fire5SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 256, 32], 0, 0.1));
var fire5SqueezeBias = dl.variable(dl.zeros([64, 8, 8, 32]));
var fire5Expand1Weights = dl.variable(dl.randomNormal([1, 1, 32, 128], 0, 0.1));
var fire5Expand1Bias = dl.variable(dl.zeros([64, 8, 8, 128]));
var fire5Expand2Weights = dl.variable(dl.randomNormal([3, 3, 32, 128], 0, 0.1));
var fire5Expand2Bias = dl.variable(dl.zeros([64, 8, 8, 128]));
// Fire 5 weights
var fire6SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 256, 48], 0, 0.1));
var fire6SqueezeBias = dl.variable(dl.zeros([64, 8, 8, 48]));
var fire6Expand1Weights = dl.variable(dl.randomNormal([1, 1, 48, 192], 0, 0.1));
var fire6Expand1Bias = dl.variable(dl.zeros([64, 8, 8, 192]));
var fire6Expand2Weights = dl.variable(dl.randomNormal([3, 3, 48, 192], 0, 0.1));
var fire6Expand2Bias = dl.variable(dl.zeros([64, 8, 8, 192]));
// Fire 6 weights
var fire7SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 384, 48], 0, 0.1));
var fire7SqueezeBias = dl.variable(dl.zeros([64, 8, 8, 48]));
var fire7Expand1Weights = dl.variable(dl.randomNormal([1, 1, 48, 192], 0, 0.1));
var fire7Expand1Bias = dl.variable(dl.zeros([64, 8, 8, 192]));
var fire7Expand2Weights = dl.variable(dl.randomNormal([3, 3, 48, 192], 0, 0.1));
var fire7Expand2Bias = dl.variable(dl.zeros([64, 8, 8, 192]));
// Fire 7 weights
var fire8SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 384, 64], 0, 0.1));
var fire8SqueezeBias = dl.variable(dl.zeros([64, 8, 8, 64]));
var fire8Expand1Weights = dl.variable(dl.randomNormal([1, 1, 64, 256], 0, 0.1));
var fire8Expand1Bias = dl.variable(dl.zeros([64, 8, 8, 256]));
var fire8Expand2Weights = dl.variable(dl.randomNormal([3, 3, 64, 256], 0, 0.1));
var fire8Expand2Bias = dl.variable(dl.zeros([64, 8, 8, 256]));
// Fire 9 weights
var fire9SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 512, 64], 0, 0.1));
var fire9SqueezeBias = dl.variable(dl.zeros([64, 4, 4, 64]));
var fire9Expand1Weights = dl.variable(dl.randomNormal([1, 1, 64, 256], 0, 0.1));
var fire9Expand1Bias = dl.variable(dl.zeros([64, 4, 4, 256]));
var fire9Expand2Weights = dl.variable(dl.randomNormal([3, 3, 64, 256], 0, 0.1));
var fire9Expand2Bias = dl.variable(dl.zeros([64, 4, 4, 256]));
// Conv 2 weights
var conv2Weights = dl.variable(dl.randomNormal([64, 1, 512, 10], 0, 0.1));
var conv2Bias = dl.variable(dl.zeros([64, 4, 4, 10]));
var SqueezeNet = /** @class */ (function () {
    function SqueezeNet() {
    }
    // private preprocessOffset = dl.tensor1d([103.939, 116.779, 123.68]);
    /**
    * Infer through SqueezeNet, assumes variables have been loaded. This does
    * standard ImageNet pre-processing before inferring through the model. This
    * method returns named activations as well as pre-softmax logits.
    *
    * @param input un-preprocessed input Array.
    * @return The pre-softmax logits.
    */
    SqueezeNet.prototype.predict = function (input) {
        return this.model(input, false).logits;
    };
    /**
    * Infer through SqueezeNet, assumes variables have been loaded. This does
    * standard ImageNet pre-processing before inferring through the model. This
    * method returns named activations as well as pre-softmax logits.
    *
    * @param input un-preprocessed input Array.
    * @param training
    * @return A requested activation and the pre-softmax logits.
    */
    SqueezeNet.prototype.model = function (input, training) {
        return dl.tidy(function () {
            log(input);
            log("Stage: preprocessing...");
            // Preprocess the input.
            // let preprocessed = dl.sub(input.asType('float32'), this.preprocessOffset) as Tensor3D;
            // log(preprocessed);
            var preprocessedInput = input.as4D(-1, IMAGE_SIZE, IMAGE_SIZE, 3);
            log("Stage: Convolution 1...");
            console.log(preprocessedInput);
            /**
             * Convolution 1
             */
            var conv1relu = preprocessedInput
                .conv2d(conv1Weights, 1, 'same')
                .add(conv1Bias)
                .relu();
            var pool1 = conv1relu.maxPool(2, 2, 'valid');
            log("Stage: Fire Module 1...");
            /**
             * Fire Module 1
             */
            var y2 = dl.tidy(function () {
                return dl.conv2d(pool1, fire2SqueezeWeights, 1, 'same')
                    .add(fire2SqueezeBias)
                    .relu();
            });
            var left2 = dl.tidy(function () {
                return dl.conv2d(y2, fire2Expand1Weights, 1, 'same')
                    .add(fire2Expand1Bias)
                    .relu();
            });
            var right2 = dl.tidy(function () {
                return dl.conv2d(y2, fire2Expand2Weights, 1, 'same')
                    .add(fire2Expand2Bias)
                    .relu();
            });
            var f2 = left2.concat(right2, 3);
            log("Stage: Fire Module 2...");
            /**
            * Fire Module 2
            */
            var y3 = dl.tidy(function () {
                return dl.conv2d(f2, fire3SqueezeWeights, 1, 'same')
                    .add(fire3SqueezeBias)
                    .relu();
            });
            var left3 = dl.tidy(function () {
                return dl.conv2d(y3, fire3Expand1Weights, 1, 'same')
                    .add(fire3Expand1Bias)
                    .relu();
            });
            var right3 = dl.tidy(function () {
                return dl.conv2d(y3, fire3Expand2Weights, 1, 'same')
                    .add(fire3Expand2Bias)
                    .relu();
            });
            var f3 = left3.concat(right3, 3);
            log("Stage: Fire Module 3...");
            /**
            * Fire Module 3
            */
            var y4 = dl.tidy(function () {
                return dl.conv2d(f3, fire4SqueezeWeights, 1, 'same')
                    .add(fire4SqueezeBias)
                    .relu();
            });
            var left4 = dl.tidy(function () {
                return dl.conv2d(y4, fire4Expand1Weights, 1, 'same')
                    .add(fire4Expand1Bias)
                    .relu();
            });
            var right4 = dl.tidy(function () {
                return dl.conv2d(y4, fire4Expand2Weights, 1, 'same')
                    .add(fire4Expand2Bias)
                    .relu();
            });
            var pool2 = dl.tidy(function () {
                var f4 = left4.concat(right4, 3);
                return f4.maxPool(2, 2, 'valid');
            });
            log("Stage: Fire Module 4...");
            /**
             * Fire Module 4
             */
            var y5 = dl.tidy(function () {
                return dl.conv2d(pool2, fire5SqueezeWeights, 1, 'same')
                    .add(fire5SqueezeBias)
                    .relu();
            });
            var left5 = dl.tidy(function () {
                return dl.conv2d(y5, fire5Expand1Weights, 1, 'same')
                    .add(fire5Expand1Bias)
                    .relu();
            });
            var right5 = dl.tidy(function () {
                return dl.conv2d(y5, fire5Expand2Weights, 1, 'same')
                    .add(fire5Expand2Bias)
                    .relu();
            });
            var f5 = left5.concat(right5, 3);
            log("Stage: Fire Module 5...");
            /**
             * Fire Module 5
             */
            var y6 = dl.tidy(function () {
                return dl.conv2d(f5, fire6SqueezeWeights, 1, 'same')
                    .add(fire6SqueezeBias)
                    .relu();
            });
            var left6 = dl.tidy(function () {
                return dl.conv2d(y6, fire6Expand1Weights, 1, 'same')
                    .add(fire6Expand1Bias)
                    .relu();
            });
            var right6 = dl.tidy(function () {
                return dl.conv2d(y6, fire6Expand2Weights, 1, 'same')
                    .add(fire6Expand2Bias)
                    .relu();
            });
            var f6 = left6.concat(right6, 3);
            log("Stage: Fire Module 6...");
            /**
             * Fire Module 6
             */
            var y7 = dl.tidy(function () {
                return dl.conv2d(f6, fire7SqueezeWeights, 1, 'same')
                    .add(fire7SqueezeBias)
                    .relu();
            });
            var left7 = dl.tidy(function () {
                return dl.conv2d(y7, fire7Expand1Weights, 1, 'same')
                    .add(fire7Expand1Bias)
                    .relu();
            });
            var right7 = dl.tidy(function () {
                return dl.conv2d(y7, fire7Expand2Weights, 1, 'same')
                    .add(fire7Expand2Bias)
                    .relu();
            });
            var f7 = left7.concat(right7, 3);
            log("Stage: Fire Module 7...");
            /**
             * Fire Module 7
             */
            var y8 = dl.tidy(function () {
                return dl.conv2d(f7, fire8SqueezeWeights, 1, 'same')
                    .add(fire8SqueezeBias)
                    .relu();
            });
            var left8 = dl.tidy(function () {
                return dl.conv2d(y8, fire8Expand1Weights, 1, 'same')
                    .add(fire8Expand1Bias)
                    .relu();
            });
            var right8 = dl.tidy(function () {
                return dl.conv2d(y8, fire8Expand2Weights, 1, 'same')
                    .add(fire8Expand2Bias)
                    .relu();
            });
            var pool3 = dl.tidy(function () {
                var f8 = left8.concat(right8, 3);
                return f8.maxPool(2, 2, 'valid');
            });
            log("Stage: Fire Module 8...");
            /**
             * Fire Module 8
             */
            var y9 = dl.tidy(function () {
                return dl.conv2d(pool3, fire9SqueezeWeights, 1, 'same')
                    .add(fire9SqueezeBias)
                    .relu();
            });
            var left9 = dl.tidy(function () {
                return dl.conv2d(y9, fire9Expand1Weights, 1, 'same')
                    .add(fire9Expand1Bias)
                    .relu();
            });
            var right9 = dl.tidy(function () {
                return dl.conv2d(y9, fire9Expand2Weights, 1, 'same')
                    .add(fire9Expand2Bias)
                    .relu();
            });
            var f9 = left9.concat(right9, 3);
            log("Stage: Convolutional Layer 2...");
            /**
             * Convolutional Layer 2
             */
            var conv10 = f9.conv2d(conv2Weights, 1, 'same')
                .add(conv2Bias);
            return {
                logits: dl.avgPool(conv10, conv10.shape[1], 1, 0).as2D(conv10.shape[0], 10),
            };
        });
    };
    // Train the model.
    SqueezeNet.prototype.train = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _this = this;
            var returnCost, _loop_1, i;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        returnCost = true;
                        _loop_1 = function (i) {
                            var cost;
                            return __generator(this, function (_a) {
                                switch (_a.label) {
                                    case 0:
                                        cost = optimizer.minimize(function () {
                                            var batch = _this.nextTrainBatch();
                                            log("iteration [" + i + "]");
                                            return SqueezeNet.loss(batch.labels, _this.model(batch.images, true).logits);
                                        }, returnCost);
                                        log("loss[" + i + "]: " + cost.dataSync());
                                        return [4 /*yield*/, dl.nextFrame()];
                                    case 1:
                                        _a.sent();
                                        return [2 /*return*/];
                                }
                            });
                        };
                        i = 0;
                        _a.label = 1;
                    case 1:
                        if (!(i < TRAIN_STEPS)) return [3 /*break*/, 4];
                        return [5 /*yield**/, _loop_1(i)];
                    case 2:
                        _a.sent();
                        _a.label = 3;
                    case 3:
                        i++;
                        return [3 /*break*/, 1];
                    case 4: return [2 /*return*/];
                }
            });
        });
    };
    /**
     * Shuffles training data and return subset of batch size
     * @return {{images: Tensor[]; labels: Tensor[]}}
     */
    SqueezeNet.prototype.nextTrainBatch = function () {
        var mapped = data.training.images.map(function (img, index) {
            return { img: img, label: data.training.labels[index] };
        });
        log(mapped);
        var shuffled = mapped.sort(function () { return .5 - Math.random(); }); // shuffle
        return { images: dl.tensor(shuffled.map(function (obj) { return obj.img; }).slice(0, BATCH_SIZE)),
            labels: dl.tensor(shuffled.map(function (obj) { return obj.label; }).slice(0, BATCH_SIZE)) };
    };
    /**
     * Loss function using softmax cross entropy
     * @param labels
     * @param logits
     * @return Tensor
     */
    SqueezeNet.loss = function (labels, logits) {
        return dl.losses.softmaxCrossEntropy(labels, logits).mean();
    };
    SqueezeNet.prototype.run_squeeze = function () {
        return __awaiter(this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        log("loading data");
                        return [4 /*yield*/, loadData()];
                    case 1:
                        _a.sent();
                        log("training!");
                        return [4 /*yield*/, this.train()];
                    case 2:
                        _a.sent();
                        return [2 /*return*/];
                }
            });
        });
    };
    return SqueezeNet;
}());
exports.SqueezeNet = SqueezeNet;
