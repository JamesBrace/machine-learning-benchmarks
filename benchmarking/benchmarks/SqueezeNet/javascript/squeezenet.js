import * as tf from '@tensorflow/tfjs';
import {CIFAR10} from "./cifar-10/cifar10-client";

/**
 * CONSTANTS
 */
// Hyper-parameters
const LEARNING_RATE = .001;
const BATCH_SIZE = 64;
const TRAIN_STEPS = 1;

// Data constants
const TRAINING_SIZE = 1001;
const TEST_SIZE = 1001;
const IMAGE_SIZE = 32;
const IMAGE_DEPTH = 3;

let data = {};

/**
 * Loads data into memory
 * @return {Promise<void>}
 */
async function loadData(training_size, test_size) {
    const cifar = new CIFAR10();

    const training = await cifar.training.get(training_size);

    const test = await cifar.test.get(test_size);

    data = {
        train: {
            images: training.map((obj) => obj.input),
            labels: training.map((obj) => obj.output),
            num_images: training.length,
        },
        test: {
            images: test.map((obj) => obj.input),
            labels: test.map((obj) => obj.output),
            num_images: test.length,
        }
    };
}


/**
 * SqueezeNet Model
 */
export class SqueezeNet {

    constructor(){
        this.model = SqueezeNet.create_model();
    }

    /**
     * Creates and compilers tensorflow model
     * @return {Model}
     */
    static create_model(){
        const optimizer = tf.train.adam(LEARNING_RATE);
        const input = tf.input({shape: [IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH]});
        const conv1 = SqueezeNet.create_conv(96, 2, input);
        const pool1 = SqueezeNet.pool(conv1);
        const fire1 = SqueezeNet.create_fire_module(pool1, 16, 64);
        const fire2 = SqueezeNet.create_fire_module(fire1, 16, 64);
        const fire3 = SqueezeNet.create_fire_module(fire2, 32, 128);
        const pool2 = SqueezeNet.pool(fire3);
        const fire4 = SqueezeNet.create_fire_module(pool2, 32, 128);
        const fire5 = SqueezeNet.create_fire_module(fire4, 48, 192);
        const fire6 = SqueezeNet.create_fire_module(fire5, 48, 192);
        const fire7 = SqueezeNet.create_fire_module(fire6, 64, 256);
        const pool3 = SqueezeNet.pool(fire7);
        const fire8 = SqueezeNet.create_fire_module(pool3, 64, 256);
        const conv2 = SqueezeNet.create_conv(10, 1, fire8);
        const output = tf.layers.globalAveragePooling2d({});
        const model = tf.model(({inputs: input, outputs: output.apply(conv2)}));

         model.compile({
            optimizer: optimizer,
            loss: 'categoricalCrossentropy',
            metrics: ['accuracy'],
        });

        return model
    }

    static create_fire_module(model, squeeze_output, expand_output){
        const squeeze = SqueezeNet.create_conv(squeeze_output, 1, model);
        return SqueezeNet.create_expand(squeeze, expand_output)
    }

    static create_conv(output, kernel_size = 1, input = null){
        const conv = tf.layers.conv2d({
            kernelSize: [kernel_size, kernel_size],
            padding: 'same',
            filters: output,
            strides: 1,
            activation: 'relu',
            kernelInitializer: 'randomNormal',
            biasInitializer: 'zeros'
        });

        return (input) ? conv.apply(input) : conv
    }

    static create_expand(model, expand_output){
        const left = SqueezeNet.create_conv(expand_output, 1, model);
        const right = SqueezeNet.create_conv(expand_output, 3, model);
        const concatLayer = tf.layers.concatenate();
        return concatLayer.apply([left, right]);
    }

    static pool(model){
        const pool = tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]});
        return pool.apply(model);
    }

    // Train the model.
    async train(set, set_size = BATCH_SIZE, train_steps = TRAIN_STEPS) {
        await this.model.fit(set.images.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH]),
                set.labels, {batchSize: BATCH_SIZE, epochs: train_steps});
        await tf.nextFrame();

    }

    // Predict the digit number from a batch of input images.
    async predict(batch, length){
        await this.model.predict(batch.images.reshape([-1 , IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH]),
            {batchSize: length});
    }

    /**
     * Shuffles training data and return subset of batch size
     * @return {{images[]; labels[]}}
     */
    nextBatch(type, size = BATCH_SIZE){

        console.log(`Info: Getting ${type} of size ${size}`);

        // let mapped = data[type].images.map((img, index) => {
        //         return {img: img, label: data[type].labels[index]}
        //     });
        //
        // console.log("Info: ", mapped);
        //
        // console.log(`Info: mapped data`);

        // const shuffled = mapped.sort(() => .5 - Math.random());// shuffle

        // console.log("Info: ", shuffled.length);
        //
        // console.log(`Info: shuffled data`);



        return {images: tf.tensor(data[type].images.slice(0, size-1)),
            labels: tf.tensor(data[type].labels.slice(0, size-1))}
    }
}

/*****************************
 *  SETUP
 ****************************/
export async function init(backend, training_size = TRAINING_SIZE, test_size = TEST_SIZE) {
    // Set backend to run on either CPU or GPU
    if(backend === 'gpu' || backend === 'cpu'){
        (backend === 'gpu') ? tf.setBackend('webgl') : tf.setBackend('cpu');
    } else {
        throw new Error(`Invalid backend parameter: ${backend}. Please specify either 'cpu' or 'gpu'`)
    }

    await loadData(training_size, test_size);
    return new SqueezeNet()
}

