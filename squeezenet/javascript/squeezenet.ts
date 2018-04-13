// // tslint:disable-next-line:max-line-length
import * as dl from 'deeplearn';
import {Scalar, Tensor1D, Tensor3D, Tensor4D} from 'deeplearn';

/*****************np
 * CONSTANTS
 ****************/
const log = console.log;

// Hyper-parameters
const LEARNING_RATE = .001;
const BATCH_SIZE = 64;
const TRAIN_STEPS = 1000;

// Data constants.
const optimizer = dl.train.adam(LEARNING_RATE);

const TRAINING_SIZE = 8000;
const TEST_SIZE = 2000;

const CIFAR10 = (<any>window).CIFAR10 || {};

let data = {training: {images: [], labels: [], num_images: 0}, test: {images: [], labels: [], num_images: 0}};

async function loadData() {
    await CIFAR10.set(TRAINING_SIZE, TEST_SIZE);

    const training = await CIFAR10.training.get(8000);
    const test = await CIFAR10.test.get(2000);

     data = {
        training: {
            images: training.map((obj:any) => obj.input),
            labels: training.map((obj:any) => obj.output),
            num_images: training.length,
        },
        test: {
            images: test.map((obj:any) => obj.input),
            labels: test.map((obj:any) => obj.output),
            num_images: test.length,
        }
    };


    log(data.training.images[0]);
    log(data.training.labels[0]);
}


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
const fire4SqueezeBias = dl.variable(dl.zeros([1, 1, 3, 32]));

const fire4Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 128], 0, 0.1));
const fire4Expand1Bias = dl.variable(dl.zeros([1, 1, 3, 128]));

const fire4Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 128], 0, 0.1));
const fire4Expand2Bias = dl.variable(dl.zeros([3, 3, 3, 128]));


// Fire 5 weights
const fire5SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 32], 0, 0.1));
const fire5SqueezeBias = dl.variable(dl.zeros([1, 1, 3, 32]));

const fire5Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 128], 0, 0.1));
const fire5Expand1Bias = dl.variable(dl.zeros([1, 1, 3, 128]));

const fire5Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 128], 0, 0.1));
const fire5Expand2Bias = dl.variable(dl.zeros([3, 3, 3, 128]));


// Fire 6 weights
const fire6SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 48], 0, 0.1));
const fire6SqueezeBias = dl.variable(dl.zeros([1, 1, 3, 48]));

const fire6Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 192], 0, 0.1));
const fire6Expand1Bias = dl.variable(dl.zeros([1, 1, 3, 192]));

const fire6Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 192], 0, 0.1));
const fire6Expand2Bias = dl.variable(dl.zeros([3, 3, 3, 192]));


// Fire 7 weights
const fire7SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 48], 0, 0.1));
const fire7SqueezeBias = dl.variable(dl.zeros([1, 1, 3, 48]));

const fire7Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 192], 0, 0.1));
const fire7Expand1Bias = dl.variable(dl.zeros([1, 1, 3, 192]));

const fire7Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 192], 0, 0.1));
const fire7Expand2Bias = dl.variable(dl.zeros([3, 3, 3, 192]));


// Fire 8 weights
const fire8SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 64], 0, 0.1));
const fire8SqueezeBias = dl.variable(dl.zeros([1, 1, 3, 64]));

const fire8Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 256], 0, 0.1));
const fire8Expand1Bias = dl.variable(dl.zeros([1, 1, 3, 256]));

const fire8Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 256], 0, 0.1));
const fire8Expand2Bias = dl.variable(dl.zeros([3, 3, 3, 256]));


// Fire 9 weights
const fire9SqueezeWeights = dl.variable(dl.randomNormal([1, 1, 3, 64], 0, 0.1));
const fire9SqueezeBias = dl.variable(dl.zeros([1, 1, 3, 64]));

const fire9Expand1Weights = dl.variable(dl.randomNormal([1, 1, 3, 256], 0, 0.1));
const fire9Expand1Bias = dl.variable(dl.zeros([1, 1, 3, 256]));

const fire9Expand2Weights = dl.variable(dl.randomNormal([3, 3, 3, 256], 0, 0.1));
const fire9Expand2Bias = dl.variable(dl.zeros([3, 3, 3, 256]));


// Conv 2 weights
const conv2Weights = dl.variable(dl.randomNormal([1, 1, 3,  1000], 0, 0.1));
const conv2Bias = dl.variable(dl.zeros([1, 1, 3,  1000]));


export class SqueezeNet {
    private preprocessOffset = dl.tensor1d([103.939, 116.779, 123.68]);


    /**
    * Infer through SqueezeNet, assumes variables have been loaded. This does
    * standard ImageNet pre-processing before inferring through the model. This
    * method returns named activations as well as pre-softmax logits.
    *
    * @param input un-preprocessed input Array.
    * @return The pre-softmax logits.
    */
    predict(input: Tensor3D): Tensor1D {
        return this.model(input, false).logits;
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
    model(input: Tensor3D, training: boolean): {logits: Tensor1D} {
        return dl.tidy(() => {

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
                return dl.conv2d(f2, fire3SqueezeWeights as Tensor4D, 1, 0)
                    .add(fire3SqueezeBias)
                    .relu() as Tensor3D;
            });

            const left3 = dl.tidy(() => {
                return dl.conv2d(y3, fire3Expand1Weights as Tensor4D, 1, 0)
                    .add(fire3Expand1Bias)
                    .relu();
            });

            const right3 = dl.tidy(() => {
                return dl.conv2d(y3, fire3Expand2Weights as Tensor4D, 1, 1)
                    .add(fire3Expand2Bias)
                    .relu();
            });

            const pool2 = dl.tidy(() => {
                const f3 = left3.concat(right3, 2) as Tensor3D;
                return f3.maxPool(3, 2, 'valid');
            });

             /**
             * Fire Module 4
             */
            const y4 = dl.tidy(() => {
                return dl.conv2d(pool2, fire4SqueezeWeights as Tensor4D, 1, 0)
                    .add(fire4SqueezeBias)
                    .relu() as Tensor3D;
            });

            const left4 = dl.tidy(() => {
                return dl.conv2d(y4, fire4Expand1Weights as Tensor4D, 1, 0)
                    .add(fire4Expand1Bias)
                    .relu();
            });

            const right4 = dl.tidy(() => {
                return dl.conv2d(y4, fire4Expand2Weights as Tensor4D, 1, 1)
                    .add(fire4Expand2Bias)
                    .relu();
            });

            const f4 = left4.concat(right4, 2) as Tensor3D;


            /**
             * Fire Module 5
             */
            const y5 = dl.tidy(() => {
                return dl.conv2d(f4, fire5SqueezeWeights as Tensor4D, 1, 0)
                    .add(fire5SqueezeBias)
                    .relu() as Tensor3D;
            });

            const left5 = dl.tidy(() => {
                return dl.conv2d(y5, fire5Expand1Weights as Tensor4D, 1, 0)
                    .add(fire5Expand1Bias)
                    .relu();
            });

            const right5 = dl.tidy(() => {
                return dl.conv2d(y5, fire5Expand2Weights as Tensor4D, 1, 1)
                    .add(fire5Expand2Bias)
                    .relu();
            });

            const pool3 = dl.tidy(() => {
                const f5 = left5.concat(right5, 2) as Tensor3D;
                return f5.maxPool(3, 2, 0);
            });

            /**
             * Fire Module 6
             */
            const y6 = dl.tidy(() => {
                return dl.conv2d(pool3, fire6SqueezeWeights as Tensor4D, 1, 0)
                    .add(fire6SqueezeBias)
                    .relu() as Tensor3D;
            });

            const left6 = dl.tidy(() => {
                return dl.conv2d(y6, fire6Expand1Weights as Tensor4D, 1, 0)
                    .add(fire6Expand1Bias)
                    .relu();
            });

            const right6 = dl.tidy(() => {
                return dl.conv2d(y6, fire6Expand2Weights as Tensor4D, 1, 1)
                    .add(fire6Expand2Bias)
                    .relu();
            });

            const f6 = left6.concat(right6, 2) as Tensor3D;

            /**
             * Fire Module 7
             */
            const y7 = dl.tidy(() => {
                return dl.conv2d(f6, fire7SqueezeWeights as Tensor4D, 1, 0)
                    .add(fire7SqueezeBias)
                    .relu() as Tensor3D;
            });

            const left7 = dl.tidy(() => {
                return dl.conv2d(y7, fire7Expand1Weights as Tensor4D, 1, 0)
                    .add(fire7Expand1Bias)
                    .relu();
            });

            const right7 = dl.tidy(() => {
                return dl.conv2d(y7, fire7Expand2Weights as Tensor4D, 1, 1)
                    .add(fire7Expand2Bias)
                    .relu();
            });

            const f7 = left7.concat(right7, 2) as Tensor3D;


            /**
             * Fire Module 8
             */
            const y8 = dl.tidy(() => {
                return dl.conv2d(f7, fire8SqueezeWeights as Tensor4D, 1, 0)
                    .add(fire8SqueezeBias)
                    .relu() as Tensor3D;
            });

            const left8 = dl.tidy(() => {
                return dl.conv2d(y8, fire8Expand1Weights as Tensor4D, 1, 0)
                    .add(fire8Expand1Bias)
                    .relu();
            });

            const right8 = dl.tidy(() => {
                return dl.conv2d(y8, fire8Expand2Weights as Tensor4D, 1, 1)
                    .add(fire8Expand2Bias)
                    .relu();
            });

            const f8 = left8.concat(right8, 2) as Tensor3D;

            /**
             * Fire Module 9
             */
            const y9 = dl.tidy(() => {
                return dl.conv2d(f8, fire9SqueezeWeights as Tensor4D, 1, 0)
                    .add(fire9SqueezeBias)
                    .relu() as Tensor3D;
            });

            const left9 = dl.tidy(() => {
                return dl.conv2d(y9, fire9Expand1Weights as Tensor4D, 1, 0)
                    .add(fire9Expand1Bias)
                    .relu();
            });

            const right9 = dl.tidy(() => {
                return dl.conv2d(y9, fire9Expand2Weights as Tensor4D, 1, 1)
                    .add(fire9Expand2Bias)
                    .relu();
            });

            const f9 = left9.concat(right9, 2) as Tensor3D;


            /**
             * Convolutaional Layer 2
             */
            const conv10 = f9.conv2d(conv2Weights as Tensor4D, 1, 0)
                .add(conv2Bias) as Tensor3D;

            return {
                logits: dl.avgPool(conv10, conv10.shape[0], 1, 0).as1D() as Tensor1D,
            };
        });
    }

    // Train the model.
    async train() {
        const returnCost = true;

        for (let i = 0; i < TRAIN_STEPS; i++) {
            const cost = optimizer.minimize(() => {
              const batch = this.nextTrainBatch();

              log(`iteration [${i}]`);
              return SqueezeNet.loss(batch.labels, this.model(batch.images, true).logits);
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
        let mapped = data.training.images.map((img: any, index: number) => {
                return {img: img, label: data.training.labels[index]}
            });

        const shuffled = mapped.sort(() => .5 - Math.random());// shuffle
        return {images: dl.tensor3d(shuffled.map((obj:any) => obj.img).slice(0, BATCH_SIZE)),
            labels: dl.tensor1d(shuffled.map((obj:any) => obj.label).slice(0, BATCH_SIZE))}
    }

    /**
     * Loss function using softmax cross entropy
     * @param labels
     * @param logits
     * @return Tensor
     */
    static loss(labels: Tensor1D, logits: Tensor1D): Scalar {
        return dl.losses.softmaxCrossEntropy(labels, logits).mean();
    }
}

async function run_squeeze(): Promise<void> {
    log("loading data");
    await loadData();
    let model = new SqueezeNet();

    log("training!");
    await model.train();
  // await test();
}

run_squeeze();
