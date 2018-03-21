const log = console.log;

/*****************************
 *  CONSTANTS
 ****************************/

// Data set sizes
const TRAINING_SIZE = 8000;
const TEST_SIZE = 2000;

// Hyper-parameters
const LEARNING_RATE = .001;
const BATCH_SIZE = 64;
const TRAIN_STEPS = 1000;

// Data constants.
const IMAGE_SIZE = 28;
const LABELS_SIZE = 10;


class MNIST_Model {

    constructor(){

        log("Backend: " + dl.getBackend());
        log(dl.memory());

        // dl.setBackend('cpu');

        /**
         * Generate the training and test data
         */
        log("Checkpoint: generating training set (size: " + TRAINING_SIZE + ") and test set (size: " + TEST_SIZE + ")");
        let set = mnist.set(TRAINING_SIZE, TEST_SIZE);

        /**
         * Assign the data to a class for better accessibility
         * @type {{training: {images, labels, num_images}, test: {images, labels, num_images: number}}}
         */
        this.data = {
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

        const optimizer = dl.train.adam(LEARNING_RATE);

        /**********************************
        * VARIABLES WE WANT TO OPTIMIZE
        **********************************/

        /**
         * Generate initial weights via random normal distribution
         * @type {{conv: {first: *, second: *}, full: {first: *, second: *}}}
         */
        this.weights = {
            conv: {
                first: MNIST_Model.weight_variable([5, 5, 1, 32]),
                second: MNIST_Model.weight_variable([5, 5, 32, 64])
            },
            full: {
                first: MNIST_Model.weight_variable([7 * 7 * 64, 1024]),
                second: MNIST_Model.weight_variable([1024, 10])
            }
        };

        /**
         * Initialize bias variable with scalar 0.1
         * @type {{conv: {first: *, second: *}, full: {first: *, second: *}}}
         */
        this.biases = {
            conv: {
                first: MNIST_Model.bias_variable([32]),
                second: MNIST_Model.bias_variable([64])
            },
            full: {
                first: MNIST_Model.bias_variable([1024]),
                second: MNIST_Model.bias_variable([10])
            }
        };


        /**
         * Train the model and then evaluate the model on the test set
         */
        this.train(() => {
          log('Successfully completed training.');
          log(`Error on test set after training: ${this.test(this.data.test.images, this.data.test.labels)}`)
        });
    }

    async train(done){
        let current_loss = 0;
        for (let iter = 0; iter < 1000; iter++) {

            // On each iteration shuffle the data
            // TODO: Implement mini-batch
            let {images, labels} = MNIST_Model.getRandom(this.data.training.images, this.data.training.labels);

            // Create tensors for images and labels to feed to optimizer
            images = dl.tensor(images);
            labels = dl.tensor(labels);

            this.optimizer.minimize(() => {

                // Feed the examples into the model
                const pred = this.predict(images);

                log("finished prediction");

                current_loss = MNIST_Model.loss(labels, pred);

                log("current loss: " + current_loss);
                return current_loss
            });

            log("Current loss after iteration " + iter + " : " + current_loss);

            // Use dl.nextFrame to not block the browser.
            await dl.nextFrame();
        }

        done();
    }

    /**
     * Given an input of images, predict the label
     * Performs eager execution
     * @param input
     * @return {*}
     */
    predict(input){
        return dl.tidy(() => {
            // Input layer
            let x = input.reshape([-1, IMAGE_SIZE, IMAGE_SIZE, 1]);
            log("Finished input reshaping");

            const strides = 2;
            const pad = 0;

            // Conv 1
            const layer1 = dl.tidy(() => {
                return x.conv2d(conv1Weights, 1, 'same')
                    .relu()
                    .maxPool([2, 2], strides, pad);
            });


            // Convolution layer #1
            let conv_1 =this.generate_first_conv_layer(x);
            log("Finished generating first conv layer");

            // Pool layer #1
            const pool_1 = dl.maxPool(conv_1, [2, 2], 2, 'same');
            log("Finished generating first pool layer");

            // Convolution layer #2
            let conv_2 = this.generate_second_conv_layer(pool_1);
            log("Finished generating second conv layer");

            // Pool layer #2
            const pool_2 = dl.maxPool(conv_2, [2, 2], 2, 'same');
            log("Finished generating second pool layer");

            // Dense layer
            const full_1 = this.generate_fully_connected_layer(pool_2);
            log("Finished first dense layer");

            // Dropout
            const drop = MNIST_Model.dropout(full_1, 0.4);
            log("Finished dropout");

            // Map the 1024 features to 10 classes, one for each digit
            const weight_fc2 = this.weights.full.second;
            const bias_fc2 = this.biases.full.second;

            // Finish training
            return dl.matMul(drop, weight_fc2) + bias_fc2;
        })
    }

    /**
     * Calculate the softmax entropy loss
     * @param labels
     * @param logits
     * @return {*|void}
     */
    static loss(labels, logits){

        console.log(labels);
        console.log(logits);
        // Calculate cross entropy
        labels.reshape([-1, 10]);
        logits.reshape([-1, 10]);

        let cross_entropy =  dl.losses.softmaxCrossEntropy(labels, logits);
        return dl.mean(cross_entropy);
    }

    /**
     * Test the learned model on a test set
     * @param xs
     * @param ys
     */
    test(xs, ys) {
      dl.tidy(() => {
        const predictedYs = this.predict(xs);
        return this.loss(ys, predictedYs)
      })
    }

    /**
     * Generate the first convolutional layer from the inputted image data
     * @param x
     * @return {*}
     */
    generate_first_conv_layer(x){

        let weight_conv1 = this.weights.conv.first;
        let bias_conv1 = this.biases.conv.first;
        return dl.tidy(() => {
            const conv_1 = dl.conv2d(x, weight_conv1, 1, 'same').add(bias_conv1);
            return conv_1.relu();
        });
    }

    /**
     * Generate the second convolutional layer from the first pool layer
     * @param pool_1
     * @return {*}
     */
    generate_second_conv_layer(pool_1){
        let weight_conv2 = this.weights.conv.second;
        let bias_conv2 = this.biases.conv.second;

        return dl.tidy(() => {
            const conv_2 = dl.conv2d(pool_1, weight_conv2, 1, 'same').add(bias_conv2);
            return conv_2.relu();
        });
    }

    /**
     * Fully connects the second pool layer
     * @param pool_2
     * @return {*}
     */
     generate_fully_connected_layer(pool_2){
        let weight_full1 = this.weights.full.first;
        let bias_full1 = this.biases.full.first;
        return dl.tidy(() => {
            const pool2_flat = dl.reshape(pool_2, [-1, 7 * 7 * 64]);
            const full = dl.matMul(pool2_flat,weight_full1).add(bias_full1);
            return full.relu()
        })
    }

    /**
     * Performs dropout based on keep prob
     * @param x
     * @param keep_prob
     * @return {*}
     */
    static dropout(x, keep_prob) {
        return dl.tidy(() => {
            if (keep_prob > 1 || keep_prob < 0) {
                throw "Keep probability must be between 0 and 1"
            }

            if (keep_prob === 1) return x;

            const uniform_tensor = dl.randomUniform(x.shape);

            const prob_tensor = dl.fill(x.shape, keep_prob);

            const random_tensor = dl.add(uniform_tensor, prob_tensor);

            const floor_tensor = dl.floor(random_tensor);

            return x.div(dl.scalar(keep_prob)).mul(floor_tensor)
        });
    }

    /**
     * Generates a weight variable of a given shape
     * @param shape
     */
    static weight_variable(shape){
        let initial = dl.randomNormal(shape, 0, 0.1);
        return dl.variable(initial)
    }

    /**
     * Generates a bias variable of a given shape
     * @param shape
     */
    static bias_variable(shape){
        let initial = dl.fill(shape, 0.1);
        return dl.variable(initial)
    }

    /**
     * Shuffles the training set
     * @param data
     * @param labels
     * @return {{images, labels}}
     */
    static getRandom(data, labels) {
        let mapped = data.map((img, index) => {
            return {img: img, label: labels[index]}
        });

        const shuffled = mapped.sort(() => .5 - Math.random());// shuffle
        return {images: shuffled.map(obj => obj.img), labels: shuffled.map(obj => obj.label)}
    }
}

new MNIST_Model();








