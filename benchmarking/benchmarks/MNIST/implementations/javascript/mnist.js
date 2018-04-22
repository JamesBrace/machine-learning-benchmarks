import * as tf from '@tensorflow/tfjs';
import {MnistData} from "./data";
import 'babel-polyfill';

const log = console.log;

/*****************************
 *  CONSTANTS
 ****************************/
// Data
const d = new MnistData();

// Hyper-parameters
const LEARNING_RATE = .001;
const BATCH_SIZE = 64;
const TRAIN_STEPS = 100;

// Data constants.
const IMAGE_SIZE = 28;
const LABELS_SIZE = 10;

export class MNIST {

    constructor(){
        this.model = {};
        this.create_model();
    }

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
    async predict(batch){
        await this.model.predict(batch.images.reshape([BATCH_SIZE, 28, 28, 1]), {batchSize: BATCH_SIZE});
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
    return new MNIST()
}
