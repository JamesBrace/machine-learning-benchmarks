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
exports.__esModule = true;
// tslint:disable-next-line:max-line-length
var dl = require("deeplearn");
// If you need to prefix the file locations with custom destination
var CIFAR10 = require("./cifar10")({ dataPath: "./data" });
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
var TRAINING_SIZE = 8000;
var TEST_SIZE = 2000;
var set = CIFAR10.set(TRAINING_SIZE, TEST_SIZE);
var data = {
    training: {
        images: set.training.map(function (obj) { return obj.input; }),
        labels: set.training.map(function (obj) { return obj.output; }),
        num_images: set.training.length
    },
    test: {
        images: set.test.map(function (obj) { return obj.input; }),
        labels: set.test.map(function (obj) { return obj.output; }),
        num_images: set.test.length
    }
};
/*****************
 * WEIGHTS
 ****************/
// Conv 1 weights
var conv1Weights = dl.variable(dl.randomNormal([7, 7, 3, 96], 0, 0.1));
var conv1Bias = dl.variable(dl.zeros([7, 7, 3, 96]));
// Fire 2 weights
var fire2SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 16], 0, 0.1));
var fire2SqueezeBias = dl.variable(dl.zeros([1, 1, 3, 16]));
var fire2Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 64], 0, 0.1));
var fire2Expand1Bias = dl.variable(dl.zeros([1, 1, 3, 64]));
var fire2Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 64], 0, 0.1));
var fire2Expand2Bias = dl.variable(dl.randomNormal([3, 3, 3, 64], 0, 0.1));
// Fire 3 weights
var fire3SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 16], 0, 0.1));
var fire3SqueezeBias = dl.variable(dl.zeros([1, 1, 3, 16]));
var fire3Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 64], 0, 0.1));
var fire3Expand1Bias = dl.variable(dl.zeros([1, 1, 3, 64]));
var fire3Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 64], 0, 0.1));
var fire3Expand2Bias = dl.variable(dl.zeros([3, 3, 3, 64]));
// Fire 4 weights
var fire4SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 32], 0, 0.1));
var fire4SqueezeBias = dl.variable(dl.zeros([1, 1, 3, 32]));
var fire4Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 128], 0, 0.1));
var fire4Expand1Bias = dl.variable(dl.zeros([1, 1, 3, 128]));
var fire4Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 128], 0, 0.1));
var fire4Expand2Bias = dl.variable(dl.zeros([3, 3, 3, 128]));
// Fire 5 weights
var fire5SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 32], 0, 0.1));
var fire5SqueezeBias = dl.variable(dl.zeros([1, 1, 3, 32]));
var fire5Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 128], 0, 0.1));
var fire5Expand1Bias = dl.variable(dl.zeros([1, 1, 3, 128]));
var fire5Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 128], 0, 0.1));
var fire5Expand2Bias = dl.variable(dl.zeros([3, 3, 3, 128]));
// Fire 6 weights
var fire6SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 48], 0, 0.1));
var fire6SqueezeBias = dl.variable(dl.zeros([1, 1, 3, 48]));
var fire6Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 192], 0, 0.1));
var fire6Expand1Bias = dl.variable(dl.zeros([1, 1, 3, 192]));
var fire6Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 192], 0, 0.1));
var fire6Expand2Bias = dl.variable(dl.zeros([3, 3, 3, 192]));
// Fire 7 weights
var fire7SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 48], 0, 0.1));
var fire7SqueezeBias = dl.variable(dl.zeros([1, 1, 3, 48]));
var fire7Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 192], 0, 0.1));
var fire7Expand1Bias = dl.variable(dl.zeros([1, 1, 3, 192]));
var fire7Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 192], 0, 0.1));
var fire7Expand2Bias = dl.variable(dl.zeros([3, 3, 3, 192]));
// Fire 8 weights
var fire8SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 64], 0, 0.1));
var fire8SqueezeBias = dl.variable(dl.zeros([1, 1, 3, 64]));
var fire8Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 256], 0, 0.1));
var fire8Expand1Bias = dl.variable(dl.zeros([1, 1, 3, 256]));
var fire8Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 256], 0, 0.1));
var fire8Expand2Bias = dl.variable(dl.zeros([3, 3, 3, 256]));
// Fire 9 weights
var fire9SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 64], 0, 0.1));
var fire9SqueezeBias = dl.variable(dl.zeros([1, 1, 3, 64]));
var fire9Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 256], 0, 0.1));
var fire9Expand1Bias = dl.variable(dl.zeros([1, 1, 3, 256]));
var fire9Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 256], 0, 0.1));
var fire9Expand2Bias = dl.variable(dl.zeros([3, 3, 3, 256]));
// Conv 2 weights
var conv2Weights = dl.variable(dl.randomNormal([1, 1, 3, 1000], 0, 0.1));
var conv2Bias = dl.variable(dl.zeros([1, 1, 3, 1000]));
var SqueezeNet = /** @class */ (function () {
    function SqueezeNet() {
        this.preprocessOffset = dl.tensor1d([103.939, 116.779, 123.68]);
    }
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
        var _this = this;
        return dl.tidy(function () {
            // Preprocess the input.
            var preprocessedInput = dl.sub(input.asType('float32'), _this.preprocessOffset);
            /**
             * Convolution 1
             */
            var conv1relu = preprocessedInput
                .conv2d(conv1Weights, 2, 0)
                .add(conv1Bias)
                .relu();
            var pool1 = conv1relu.maxPool(3, 2, 0);
            /**
             * Fire Module 2
             */
            var y2 = dl.tidy(function () {
                return dl.conv2d(pool1, fire2SqueezeWeights, 1, 0)
                    .add(fire2SqueezeBias)
                    .relu();
            });
            var left2 = dl.tidy(function () {
                return dl.conv2d(y2, fire2Expand1Weights, 1, 0)
                    .add(fire2Expand1Bias)
                    .relu();
            });
            var right2 = dl.tidy(function () {
                return dl.conv2d(y2, fire2Expand2Weights, 1, 1)
                    .add(fire2Expand2Bias)
                    .relu();
            });
            var f2 = left2.concat(right2, 2);
            /**
            * Fire Module 3
            */
            var y3 = dl.tidy(function () {
                return dl.conv2d(f2, fire3SqueezeWeights, 1, 0)
                    .add(fire3SqueezeBias)
                    .relu();
            });
            var left3 = dl.tidy(function () {
                return dl.conv2d(y3, fire3Expand1Weights, 1, 0)
                    .add(fire3Expand1Bias)
                    .relu();
            });
            var right3 = dl.tidy(function () {
                return dl.conv2d(y3, fire3Expand2Weights, 1, 1)
                    .add(fire3Expand2Bias)
                    .relu();
            });
            var pool2 = dl.tidy(function () {
                var f3 = left3.concat(right3, 2);
                return f3.maxPool(3, 2, 'valid');
            });
            /**
            * Fire Module 4
            */
            var y4 = dl.tidy(function () {
                return dl.conv2d(pool2, fire4SqueezeWeights, 1, 0)
                    .add(fire4SqueezeBias)
                    .relu();
            });
            var left4 = dl.tidy(function () {
                return dl.conv2d(y4, fire4Expand1Weights, 1, 0)
                    .add(fire4Expand1Bias)
                    .relu();
            });
            var right4 = dl.tidy(function () {
                return dl.conv2d(y4, fire4Expand2Weights, 1, 1)
                    .add(fire4Expand2Bias)
                    .relu();
            });
            var f4 = left4.concat(right4, 2);
            /**
             * Fire Module 5
             */
            var y5 = dl.tidy(function () {
                return dl.conv2d(f4, fire5SqueezeWeights, 1, 0)
                    .add(fire5SqueezeBias)
                    .relu();
            });
            var left5 = dl.tidy(function () {
                return dl.conv2d(y5, fire5Expand1Weights, 1, 0)
                    .add(fire5Expand1Bias)
                    .relu();
            });
            var right5 = dl.tidy(function () {
                return dl.conv2d(y5, fire5Expand2Weights, 1, 1)
                    .add(fire5Expand2Bias)
                    .relu();
            });
            var pool3 = dl.tidy(function () {
                var f5 = left5.concat(right5, 2);
                return f5.maxPool(3, 2, 0);
            });
            /**
             * Fire Module 6
             */
            var y6 = dl.tidy(function () {
                return dl.conv2d(pool3, fire6SqueezeWeights, 1, 0)
                    .add(fire6SqueezeBias)
                    .relu();
            });
            var left6 = dl.tidy(function () {
                return dl.conv2d(y6, fire6Expand1Weights, 1, 0)
                    .add(fire6Expand1Bias)
                    .relu();
            });
            var right6 = dl.tidy(function () {
                return dl.conv2d(y6, fire6Expand2Weights, 1, 1)
                    .add(fire6Expand2Bias)
                    .relu();
            });
            var f6 = left6.concat(right6, 2);
            /**
             * Fire Module 7
             */
            var y7 = dl.tidy(function () {
                return dl.conv2d(f6, fire7SqueezeWeights, 1, 0)
                    .add(fire7SqueezeBias)
                    .relu();
            });
            var left7 = dl.tidy(function () {
                return dl.conv2d(y7, fire7Expand1Weights, 1, 0)
                    .add(fire7Expand1Bias)
                    .relu();
            });
            var right7 = dl.tidy(function () {
                return dl.conv2d(y7, fire7Expand2Weights, 1, 1)
                    .add(fire7Expand2Bias)
                    .relu();
            });
            var f7 = left7.concat(right7, 2);
            /**
             * Fire Module 8
             */
            var y8 = dl.tidy(function () {
                return dl.conv2d(f7, fire8SqueezeWeights, 1, 0)
                    .add(fire8SqueezeBias)
                    .relu();
            });
            var left8 = dl.tidy(function () {
                return dl.conv2d(y8, fire8Expand1Weights, 1, 0)
                    .add(fire8Expand1Bias)
                    .relu();
            });
            var right8 = dl.tidy(function () {
                return dl.conv2d(y8, fire8Expand2Weights, 1, 1)
                    .add(fire8Expand2Bias)
                    .relu();
            });
            var f8 = left8.concat(right8, 2);
            /**
             * Fire Module 9
             */
            var y9 = dl.tidy(function () {
                return dl.conv2d(f8, fire9SqueezeWeights, 1, 0)
                    .add(fire9SqueezeBias)
                    .relu();
            });
            var left9 = dl.tidy(function () {
                return dl.conv2d(y9, fire9Expand1Weights, 1, 0)
                    .add(fire9Expand1Bias)
                    .relu();
            });
            var right9 = dl.tidy(function () {
                return dl.conv2d(y9, fire9Expand2Weights, 1, 1)
                    .add(fire9Expand2Bias)
                    .relu();
            });
            var f9 = left9.concat(right9, 2);
            /**
             * Convolutaional Layer 2
             */
            var conv10 = f9.conv2d(conv2Weights, 1, 0)
                .add(conv2Bias);
            return {
                logits: dl.avgPool(conv10, conv10.shape[0], 1, 0).as1D()
            };
        });
    };
    // Train the model.
    SqueezeNet.prototype.train = function () {
        return __awaiter(this, void 0, void 0, function () {
            var _this = this;
            var returnCost, i, cost;
            return __generator(this, function (_a) {
                switch (_a.label) {
                    case 0:
                        returnCost = true;
                        i = 0;
                        _a.label = 1;
                    case 1:
                        if (!(i < TRAIN_STEPS)) return [3 /*break*/, 4];
                        cost = optimizer.minimize(function () {
                            var batch = _this.nextTrainBatch();
                            return SqueezeNet.loss(batch.labels, _this.model(batch.images, true));
                        }, returnCost);
                        log("loss[" + i + "]: " + cost.dataSync());
                        return [4 /*yield*/, dl.nextFrame()];
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
    return SqueezeNet;
}());
exports.SqueezeNet = SqueezeNet;
function run_squeeze() {
    return __awaiter(this, void 0, void 0, function () {
        var model;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    model = new SqueezeNet();
                    return [4 /*yield*/, model.train()];
                case 1:
                    _a.sent();
                    return [2 /*return*/];
            }
        });
    });
}
run_squeeze();
