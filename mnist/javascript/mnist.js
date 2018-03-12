const log = console.log;

/**
 * Size of training set
 * @type {number}
 */
const training_size = 8000;


/**
 * Size of test set
 * @type {number}
 */
const test_size = 2000;


class MNIST_Model {

    constructor(){

        log("Backend: " + dl.getBackend());
        log(dl.memory());


        log("Checkpoint: generating training set (size: " + training_size + ") and test set (size: " + test_size + ")");
        let set = mnist.set(training_size, test_size);

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

        this.optimizer = dl.train.adam(0.0001);

        this.train(() => {
          log('Successfully completed training.');
          log(`Error on test set after training: ${this.test(this.data.test.images, this.data.test.labels)}`)
        });
    }

    async train(done){
        let current_loss = 0;
        for (let iter = 0; iter < 100; iter++) {
            let {images, labels} = MNIST_Model.getRandom(this.data.training.images, this.data.training.labels);

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
     * @param input
     * @return {*}
     */
    predict(input){
        return dl.tidy(() => {
            // Input layer
            let x = input.reshape([-1, 28, 28, 1]);
            log("Finished input reshaping");

            // Convolution layer #1
            let conv_1 =this.generate_first_conv_layer(x);
            log("Finished generating first conv layer");
            conv_1.print();

            // Pool layer #1
            const pool_1 = dl.maxPool(conv_1, [2, 2], 2, 'same');
            log("Finished generating first pool layer");
            pool_1.print();

            // Convolution layer #2
            let conv_2 = this.generate_second_conv_layer(pool_1);
            log("Finished generating second conv layer");
            conv_2.print();

            // Pool layer #2
            const pool_2 = dl.maxPool(conv_2, [2, 2], 2, 'same');
            log("Finished generating second pool layer");
            pool_2.print();

            // Dense layer
            const full_1 = this.generate_fully_connected_layer(pool_2);
            log("Finished first dense layer");
            full_1.print();

            // Dropout
            const drop = MNIST_Model.dropout(full_1, 0.4);
            log("Finished dropout");
            drop.print();

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
        return dl.tidy(() => {
            let weight_conv1 = this.weights.conv.first;
            let bias_conv1 = this.biases.conv.first;
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
        return dl.tidy(() => {
            let weight_conv2 = this.weights.conv.second;
            let bias_conv2 = this.biases.conv.second;
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
         return dl.tidy(() => {
            let weight_full1 = this.weights.full.first;
            let bias_full1 = this.biases.full.first;
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
        return dl.tidy(() => {
            let initial = dl.randomNormal(shape);
            return dl.variable(initial)
        });
    }

    /**
     * Generates a bias variable of a given shape
     * @param shape
     */
    static bias_variable(shape){
        return dl.tidy(() => {
            let initial = dl.fill(shape, 0.1);
            return dl.variable(initial)
        });
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








