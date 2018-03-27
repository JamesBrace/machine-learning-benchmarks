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
// tslint:disable-next-line:max-line-length
import * as dl from 'deeplearn';
import {Tensor1D, Tensor3D, Tensor4D} from 'deeplearn';
import * as model_util from './util';
import {IMAGENET_CLASSES} from './imagenet_classes';

/*****************
 * CONSTANTS
 ****************/
const log = console.log;

// Hyper-parameters
const LEARNING_RATE = .001;
const BATCH_SIZE = 64;
const TRAIN_STEPS = 1000;

// Data constants.
const IMAGE_SIZE = 224 * 224 * 3;
const LABELS_SIZE = 1000;
const optimizer = dl.train.adam(LEARNING_RATE);

const data = {
    training: {
        images: [],
        labels: [],
        num_images: [],
    },
    test: {
        images: [],
        labels: [],
        num_images: [],
    }
};


/*****************
 * WEIGHTS
 ****************/
// Conv 1 weights
const conv1Weights = dl.variable(dl.randomNormal([7, 7, 3,  96], 0, 0.1));
const conv1Bias = dl.variable(dl.zeros([7, 7, 3,  96]));

// Fire 2 weights
const fire2SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 16], 0, 0.1));
const fire2SqueezeBias = dl.variable(dl.zeros([1, 1, 3, 16]));

const fire2Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 64], 0, 0.1));
const fire2Expand1Bias = dl.variable(dl.zeros([1, 1, 3, 64]));

const fire2Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 64], 0, 0.1));
const fire2Expand2Bias = dl.variable(dl.randomNormal([3, 3, 3, 64], 0, 0.1));

// Fire 3 weights
const fire3SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 16], 0, 0.1));
const fire3SqueezeBias = dl.variable(dl.zeros([1, 1, 3, 16]));

const fire3Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 64], 0, 0.1));
const fire3Expand1Bias = dl.variable(dl.zeros([1, 1, 3, 64]));

const fire3Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 64], 0, 0.1));
const fire3Expand2Bias = dl.variable(dl.zeros([3, 3, 3, 64]));


// Fire 4 weights
const fire4SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 32], 0, 0.1));
const fire4Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 128], 0, 0.1));
const fire4Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 128], 0, 0.1));

// Fire 5 weights
const fire5SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 32], 0, 0.1));
const fire5Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 128], 0, 0.1));
const fire5Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 128], 0, 0.1));

// Fire 6 weights
const fire6SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 48], 0, 0.1));
const fire6Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 192], 0, 0.1));
const fire6Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 192], 0, 0.1));

// Fire 7 weights
const fire7SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 48], 0, 0.1));
const fire7Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 192], 0, 0.1));
const fire7Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 192], 0, 0.1));

// Fire 8 weights
const fire8SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 64], 0, 0.1));
const fire8Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 256], 0, 0.1));
const fire8Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 256], 0, 0.1));

// Fire 9 weights
const fire9SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 64], 0, 0.1));
const fire9Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 256], 0, 0.1));
const fire9Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 256], 0, 0.1));

// Conv 2 weights
const conv2Weights = dl.variable(dl.randomNormal([1, 1, 3,  1000], 0, 0.1));


export class SqueezeNet {
    private variables: {[varName: string]: dl.Tensor};
    private preprocessOffset = dl.tensor1d([103.939, 116.779, 123.68]);


    /**
    * Loads necessary variables for SqueezeNet.
    */
    async load(): Promise<void> {
        const checkpointLoader = new dl.CheckpointLoader(GOOGLE_CLOUD_STORAGE_DIR + 'squeezenet1_1/');
        this.variables = await checkpointLoader.getAllVariables();
    }

    /**
    * Infer through SqueezeNet, assumes variables have been loaded. This does
    * standard ImageNet pre-processing before inferring through the model. This
    * method returns named activations as well as pre-softmax logits.
    *
    * @param input un-preprocessed input Array.
    * @return The pre-softmax logits.
    */
    predict(input: Tensor3D): Tensor1D {
        return this.predictWithActivation(input).logits;
    }

    /**
    * Infer through SqueezeNet, assumes variables have been loaded. This does
    * standard ImageNet pre-processing before inferring through the model. This
    * method returns named activations as well as pre-softmax logits.
    *
    * @param input un-preprocessed input Array.
    * @param training
    * @return A requested activation and the pre-softmax logits.
    */
    model(input: Tensor3D, training: boolean): {logits: Tensor1D, activation: Tensor3D} {
        return dl.tidy(() => {
            let activation: Tensor3D;

            // Preprocess the input.
            const preprocessedInput = dl.sub(input.asType('float32'), this.preprocessOffset) as Tensor3D;

            /**
             * Convolution 1
             */
            const conv1relu = preprocessedInput
                  .conv2d(conv1Weights as Tensor4D, 2, 0)
                  .add(conv1Bias as Tensor1D)
                  .relu() as Tensor3D;

            const pool1 = conv1relu.maxPool(3, 2, 0);

            /**
             * Fire Module 2
             */
            const y2 = dl.tidy(() => {
                return dl.conv2d(pool1, fire2SqueezeWeights as Tensor4D, 1, 0)
                    .add(fire2SqueezeBias)
                    .relu() as Tensor3D;
            });

            const left2 = dl.tidy(() => {
                return dl.conv2d(y2, fire2Expand1Weights as Tensor4D, 1, 0)
                    .add(fire2Expand1Bias)
                    .relu();
            });

            const right2 = dl.tidy(() => {
                return dl.conv2d(y2, fire2Expand2Weights as Tensor4D, 1, 1)
                    .add(fire2Expand2Bias)
                    .relu();
            });

            const f2 = left2.concat(right2, 2) as Tensor3D;

             /**
             * Fire Module 3
             */
            const y3 = dl.tidy(() => {
                return dl.conv2d(f2, fire2SqueezeWeights as Tensor4D, 1, 0)
                    .add(fire3SqueezeBias)
                    .relu() as Tensor3D;
            });

            const left3 = dl.tidy(() => {
                return dl.conv2d(y3, fire2Expand1Weights as Tensor4D, 1, 0)
                    .add(fire3Expand1Bias)
                    .relu();
            });

            const right3 = dl.tidy(() => {
                return dl.conv2d(y3, fire2Expand2Weights as Tensor4D, 1, 1)
                    .add(fire3Expand2Bias)
                    .relu();
            });

            const pool2 = dl.tidy(() => {
                const f3 = left3.concat(right3, 2) as Tensor3D;
                return f3.maxPool(3, 2, 'valid');
            });

             /**
             * Fire Module 3
             */
            const y3 = dl.tidy(() => {
                return dl.conv2d(f2, fire2SqueezeWeights as Tensor4D, 1, 0)
                    .add(fire3SqueezeBias)
                    .relu() as Tensor3D;
            });

            const left3 = dl.tidy(() => {
                return dl.conv2d(y3, fire2Expand1Weights as Tensor4D, 1, 0)
                    .add(fire3Expand1Bias)
                    .relu();
            });

            const right3 = dl.tidy(() => {
                return dl.conv2d(y3, fire2Expand2Weights as Tensor4D, 1, 1)
                    .add(fire3Expand2Bias)
                    .relu();
            });

            const pool2 = dl.tidy(() => {
                const f3 = left3.concat(right3, 2) as Tensor3D;
                return f3.maxPool(3, 2, 'valid');
            });




            const fire5 = this.fireModule(fire4, 5);

            const pool3 = fire5.maxPool(3, 2, 0);

            const fire6 = this.fireModule(pool3, 6);

            const fire7 = this.fireModule(fire6, 7);

            const fire8 = this.fireModule(fire7, 8);

            const fire9 = this.fireModule(fire8, 9);

            const conv10 = fire9.conv2d(this.variables['conv10_W:0'] as Tensor4D, 1, 0)
                .add(this.variables['conv10_b:0']) as Tensor3D;

            return {
                logits: dl.avgPool(conv10, conv10.shape[0], 1, 0).as1D() as Tensor1D,
                activation: activation as Tensor3D
            };
        });
    }

    private fireModule(input: Tensor3D, fireId: number) {
        const y =
            dl.conv2d(
                  input, this.variables[`fire${fireId}/squeeze1x1_W:0`] as Tensor4D,
                  1, 0)
                .add(this.variables[`fire${fireId}/squeeze1x1_b:0`])
                .relu() as Tensor3D;

        const left =
            dl.conv2d(
                  y, this.variables[`fire${fireId}/expand1x1_W:0`] as Tensor4D, 1,
                  0)
                .add(this.variables[`fire${fireId}/expand1x1_b:0`])
                .relu();

        const right =
            dl.conv2d(
                  y, this.variables[`fire${fireId}/expand3x3_W:0`] as Tensor4D, 1,
                  1)
                .add(this.variables[`fire${fireId}/expand3x3_b:0`])
                .relu();

        return left.concat(right, 2) as Tensor3D;
    }

  /**
   * Get the topK classes for pre-softmax logits. Returns a map of className
   * to softmax normalized probability.
   *
   * @param logits Pre-softmax logits array.
   * @param topK How many top classes to return.
   */
    async getTopKClasses(logits: Tensor1D, topK: number): Promise<{[className: string]: number}> {

        const predictions = dl.tidy(() => {
            return dl.softmax(logits).asType('float32');
        });

        const topk = model_util.topK(await predictions.data() as Float32Array, topK);

        predictions.dispose();

        const topkIndices = topk.indices;
        const topkValues = topk.values;

        const topClassesToProbability: {[className: string]: number} = {};

        for (let i = 0; i < topkIndices.length; i++) {
            topClassesToProbability[IMAGENET_CLASSES[topkIndices[i]]] = topkValues[i];
        }

        return topClassesToProbability;
    }

    dispose() {
        this.preprocessOffset.dispose();

        for (const varName in this.variables) {
            this.variables[varName].dispose();
        }
    }

    // Train the model.
    async train() {
        const returnCost = true;

        for (let i = 0; i < TRAIN_STEPS; i++) {
            const cost = optimizer.minimize(() => {
              const batch = this.nextTrainBatch();
              return SqueezeNet.loss(batch.labels, this.model(batch.images, true));
            }, returnCost);

            log(`loss[${i}]: ${cost.dataSync()}`);

            await dl.nextFrame();
        }
    }

    /**
     * Shuffles training data and return subset of batch size
     * @return {{images: Tensor[]; labels: Tensor[]}}
     */
    nextTrainBatch(){
        let mapped = data.training.images.map((img, index) => {
                return {img: img, label: data.training.labels[index]}
            });

        const shuffled = mapped.sort(() => .5 - Math.random());// shuffle
        return {images: dl.tensor(shuffled.map(obj => obj.img).slice(0, BATCH_SIZE)),
            labels: dl.tensor(shuffled.map(obj => obj.label).slice(0, BATCH_SIZE))}
    }

    /**
     * Loss function using softmax cross entropy
     * @param labels
     * @param logits
     * @return Tensor
     */
    static loss(labels, logits) {
        return dl.losses.softmaxCrossEntropy(labels, logits).mean();
    }
}
