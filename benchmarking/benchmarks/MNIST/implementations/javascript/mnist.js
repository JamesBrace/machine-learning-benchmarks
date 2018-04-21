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
import {MnistData} from "./data";
import 'babel-polyfill';

/*****************************
 *  CONSTANTS
 ****************************/
// Data
const d = new MnistData();

// Hyper-parameters
const LEARNING_RATE = .001;
const BATCH_SIZE = 64;
const TRAIN_STEPS = 1000;

// Data constants.
const IMAGE_SIZE = 28;
const LABELS_SIZE = 10;

export class MNIST {

    model = {};

    // Create the model and assigns it to the global model property
    create_model() {
        const optimizer = tf.train.adam(LEARNING_RATE);
        const model = tf.sequential();

        model.add(tf.layers.conv2d({
            inputShape: [IMAGE_SIZE, IMAGE_SIZE, 1],
            kernelSize: 5,
            filters: 32,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'randomNormal',
            biasInitializer: 'zeros'
        }));

        model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

        model.add(tf.layers.conv2d({
            kernelSize: 5,
            filters: 64,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'randomNormal',
            biasInitializer: 'zeros'
        }));

        model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

        model.add(tf.layers.flatten());

        model.add(tf.layers.dense({
            units: LABELS_SIZE, kernelInitializer: 'randomNormal',
            biasInitializer: 'zeros', activation: 'softmax'
        }));

        model.compile({
            optimizer: optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        });

        this.model = model;
    }

    // Train the model.
    async train() {
        for (let i = 0; i < TRAIN_STEPS; i++) {
            const batch = nextBatch('train');
            await this.model.fit(batch.images.reshape([BATCH_SIZE, 28, 28, 1]), batch.labels, {batchSize: BATCH_SIZE, epochs: 1});
            await tf.nextFrame();
        }
    }

    // Predict the digit number from a batch of input images.
    predict(){
        tf.tidy(() => {
            this.model.predict(batch.images.reshape([-1, 28, 28, 1]));
        });
    }
}


/*****************************
 * HELPERS
 ****************************/
// Gets the next shuffled training batch
export function nextBatch(type, batch_size = BATCH_SIZE) {
    return (type === 'train') ? d.nextTrainBatch(batch_size) : d.nextTestBatch(batch_size);
}

// Sets the data in the data.js file
export async function set_data(){
    await d.load()
}

/*****************************
 *  SETUP
 ****************************/
export async function setup(backend) {
    // Set backend to run on either CPU or GPU
    if(backend === 'gpu' || backend === 'cpu'){
        (backend === 'gpu') ? tf.setBackend('webgl') : tf.setBackend('cpu');
    } else {
        throw new Error(`Invalid backend parameter: ${backend}. Please specify either 'cpu' or 'gpu'`)
    }

    await set_data();
    let m = new MNIST();
    m.create_model();
    return m
}
