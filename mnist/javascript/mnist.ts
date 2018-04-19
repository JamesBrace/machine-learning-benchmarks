/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2018 James Brace
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

import * as tf from '@tensorflow/tfjs';
import 'babel-polyfill';
import {AdamOptimizer, Tensor, Tensor1D, Tensor2D, Tensor4D, Scalar, Variable} from '@tensorflow/tfjs';
const log = console.log;

/*****************************
 *  CONSTANTS
 ****************************/

// Data set sizes
const TRAINING_SIZE: number = 8000;
const TEST_SIZE: number = 2000;
const set = mnist.set(TRAINING_SIZE, TEST_SIZE);

log(set);

// MNIST Data
const data = {
    training: {
        images: set.training.map(obj => obj.input),
        labels: set.training.map(obj => obj.output),
        num_images: set.training.length,
    },
    test: {
        images: set.test.map(obj => obj.input),
        labels: set.test.map(obj => obj.output),
        num_images: set.test.length,
    }
};

// Hyper-parameters
const LEARNING_RATE: number = .001;
const BATCH_SIZE: number = 64;
const TRAIN_STEPS: number = 1000;

// Data constants.
const IMAGE_SIZE: number = 28;
const LABELS_SIZE: number = 10;
const optimizer: AdamOptimizer = tf.train.adam(LEARNING_RATE);

/*****************************
 *  WEIGHTS
 ****************************/

// Variables that we want to optimize
const conv1OutputDepth: number = 32;
const conv1Weights: Variable = tf.variable(tf.randomNormal([5, 5, 1, conv1OutputDepth], 0, 0.1));

const conv2InputDepth: number = conv1OutputDepth;
const conv2OutputDepth: number = 64;
const conv2Weights: Variable = tf.variable(tf.randomNormal([5, 5, conv2InputDepth, conv2OutputDepth], 0, 0.1));

const fullyConnectedWeights1: Variable = tf.variable(tf.randomNormal([7 * 7 * conv2OutputDepth, 1024], 0, 0.1));
const fullyConnectedBias1: Variable = tf.variable(tf.zeros([1024]));

const fullyConnectedWeights2: Variable  = tf.variable(tf.randomNormal([1024, LABELS_SIZE], 0, 0.1));
const fullyConnectedBias2: Variable  = tf.variable(tf.zeros([10]));

// Loss function
function loss(labels: Tensor2D, logits: Tensor2D): Tensor1D {
    return tf.losses.softmaxCrossEntropy(labels, logits).mean();
}

/*****************************
 *  MODEL
 ****************************/
function model(inputXs: Tensor, training: boolean): Tensor2D {
    const xs = inputXs.as4D(-1, IMAGE_SIZE, IMAGE_SIZE, 1);

    const strides: number = 2;
    const keep_prob: number = 0.4;

    // Conv 1
    const layer1: Tensor4D = tf.tidy(() => {
        return xs.conv2d(conv1Weights, 1, 'same')
            .relu()
            .maxPool([2, 2], strides, 'same');
    });

    // Conv 2
    const layer2: Tensor4D = tf.tidy(() => {
        return layer1.conv2d(conv2Weights, 1, 'same')
            .relu()
            .maxPool([2, 2], strides, 'same');
    });

    // Dense layer
    const full = tf.tidy(() => {
        return layer2.as2D(-1, 7 * 7 * 64)
            .matMul(fullyConnectedWeights1)
            .add(fullyConnectedBias1)
            .relu();
    }) as Tensor2D;

    if(training){
        // Dropout
        const dropout: Tensor2D = tf.tidy(() => {
            if (keep_prob > 1 || keep_prob < 0) {
                throw "Keep probability must be between 0 and 1"
            }

            if (keep_prob === 1) return full;

            const uniform_tensor: Tensor2D = tf.randomUniform(full.shape);
            const prob_tensor: Tensor2D = tf.fill(full.shape, keep_prob);
            const random_tensor: Tensor2D = tf.add(uniform_tensor, prob_tensor);
            const floor_tensor: Tensor2D = tf.floor(random_tensor);

            return full.div(tf.scalar(keep_prob)).mul(floor_tensor)
        });

        return tf.matMul(dropout, fullyConnectedWeights2).add(fullyConnectedBias2);
    }

    return tf.matMul(full, fullyConnectedWeights2).add(fullyConnectedBias2);
}

/*****************************
 * HELPERS
 ****************************/

// Gets the next shuffled training batch
function nextBatch(type: string, batch_size = BATCH_SIZE): {images: Tensor, labels: Tensor} {
    let mapped = data[type].images.map((img, index) => {
            return {img: img, label: data.training.labels[index]}
        });

    const shuffled = mapped.sort(() => .5 - Math.random());// shuffle
    return {images: tf.tensor(shuffled.map(obj => obj.img).slice(0, batch_size)),
        labels: tf.tensor(shuffled.map(obj => obj.label).slice(0, batch_size))}
}

// Train the model.
async function train(): Promise<{cumul: number, model: number}> {
    let model_time: number = 0;
    const returnCost: boolean = true;

    let cumul_time_start = performance.now();
    for (let i = 0; i < TRAIN_STEPS; i++) {
        const cost: Scalar = optimizer.minimize(() => {
            const batch = nextBatch('training');
            let start_model = performance.now();
            let logits: Tensor2D = model(batch.images, true);
            let end_model = performance.now();
            model_time += end_model - start_model;
            return loss(batch.labels, logits);
        }, returnCost);

        // log(`loss[${i}]: ${cost.dataSync()}`);

        await tf.nextFrame();
    }

    let cumul_time_end = performance.now();
    return {cumul: cumul_time_end - cumul_time_start, model: model_time/TRAIN_STEPS}
}

// Predict the digit number from a batch of input images.
function predict(x){

    let times = {cumul: 0, model: 0};

    const pred = tf.tidy(() => {
        const axis = 1;

        let start = performance.now();
        let pred = model(x, false).argMax(axis);
        let end = performance.now();

        times.cumul = times.model = end - start;

        return pred
    });

  return {times: times, logits: Array.from(pred.dataSync())} ;
}

// Given a logits or label vector, return the class indices.
function classesFromLabel(y) {
  const axis = 1;
  const pred = y.argMax(axis);
  return Array.from(pred.dataSync());
}

async function test(): Promise<{cumul: number, model: number}> {
    const batch = nextBatch('test', 1);
    const predictions = predict(batch.images);

    // const labels = classesFromLabel(batch.labels);
    // log(loss(labels, predictions.logits))

    return {cumul: predictions.times.cumul, model: predictions.times.model}
}


/*****************************
 *  DRIVER
 ****************************/
export async function run_mnist(backend, mode) {

    // Set backend to run on either CPU or GPU
    if(backend === 'gpu' || backend === 'cpu'){
        (backend === 'gpu') ? tf.setBackend('webgl') : tf.setBackend('cpu');
    } else {
        throw new Error(`Invalid backend parameter: ${backend}. Please specify either 'cpu' or 'gpu'`)
    }


    let times = (mode === 'train') ? await train() : await test();
    let images_per_sec = (mode === 'train') ? times.model/ BATCH_SIZE * TRAIN_STEPS : times.model;
    console.log(`Backend: ${backend}, Cumulative time: ${times.cumul}, Model time: ${times.model}, Images/sec: ${images_per_sec}`);

    return {
        status: 1,
        options: `run_mnist('${backend}', '${mode}')`,
        time: {cumulative: times.cumul, model: times.model, images_per_sec: images_per_sec}
    }
}

run_mnist('gpu', 'train');
