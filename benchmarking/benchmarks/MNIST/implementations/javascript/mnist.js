import * as tf from '@tensorflow/tfjs';
import {MnistData} from "./data";

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
const TRAIN_SIZE = 50000;
const TEST_SIZE = 10000;

export class MNIST {

    constructor(){
        this.model = MNIST.create_model();
    }

    // Create the model and assigns it to the global model property
    static create_model() {
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

        return model
    }

    // Train the model.
    async train(batch, train_steps = TRAIN_STEPS) {
        for (let i = 0; i < train_steps; i++) {
            await this.model.fit(batch.images.reshape([TRAIN_SIZE, 28, 28, 1]), batch.labels, {batchSize: BATCH_SIZE, epochs: 1});
            await tf.nextFrame();
        }
    }

    // Predict the digit number from a batch of input images.
    async predict(batch){
        await this.model.predict(batch.images.reshape([TEST_SIZE, 28, 28, 1]), {batchSize: BATCH_SIZE});
    }
}


/*****************************
 * HELPERS
 ****************************/
// Gets the next shuffled training batch
export function nextBatch(type) {
    return (type === 'train') ? d.nextTrainBatch(55000) : d.nextTestBatch(10000);
}

// Sets the data in the data.js file
async function set_data(){
    await d.load()
}

/*****************************
 *  SETUP
 ****************************/
export async function init(backend) {
    // Set backend to run on either CPU or GPU
    if(backend === 'gpu' || backend === 'cpu'){
        (backend === 'gpu') ? tf.setBackend('webgl') : tf.setBackend('cpu');
    } else {
        throw new Error(`Invalid backend parameter: ${backend}. Please specify either 'cpu' or 'gpu'`)
    }

    await set_data();
    return new MNIST()
}
