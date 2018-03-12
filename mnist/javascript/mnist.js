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

        log("Checkpoint: generating training set (size: " + training_size + ") and test set (size: " + test_size + ")");

        let set = mnist.set(training_size, test_size);

        this.data = {
            training: {
                images: set.training.map(obj => obj.input),
                labels: this.extract_labels(set.training),
                num_images: set.training.length,
            },
            test: {
                images: set.test.map(obj => obj.input),
                labels: this.extract_labels(set.test),
                num_images: set.test.length,
            }
        };

        log("Checkpoint: Finished pre-processing");
        log("Checkpoint: Starting training");

        let conv = this.create_NN();

        // Calculate cross entropy
        let cross_entropy =  dl.losses.softmaxCrossEntropy(this.data.training.labels, conv);
        cross_entropy = dl.mean(cross_entropy);

        // Adam optimization
        let train_step = dl.train.adam(0.0001).minimize(cross_entropy);

        

    }

    extract_labels(set){
        let labels = [];
        set.map(obj => {
            obj.output.map((l, i) => {
                if(l === 1) labels.push(i)
            })
        });
        return labels
    }

    create_NN(){

        // Input layer
        let x = dl.tensor2d(this.data.training.images);
        x = x.reshape([-1, 28, 28, 1]);
        log("Checkpoint: generated input layer");

        // Convolution layer #1
        let conv_1 = MNIST_Model.generate_first_conv_layer(x);
        log("Checkpoint: generated first convolutional layer");

        // Pool layer #1
        const pool_1 = dl.maxPool(conv_1, [2, 2], 2, 'same');
        log("Checkpoint: generated first max pool layer");

        // Convolution layer #2
        let conv_2 = MNIST_Model.generate_second_conv_layer(pool_1);
        log("Checkpoint: generated second convolutional layer");

        // Pool layer #2
        const pool_2 = dl.maxPool(conv_2, [2, 2], 2, 'same');
        log("Checkpoint: generated second max pool layer");

        // Dense layer
        const full_1 = MNIST_Model.generate_fully_connected_layer(pool_2);
        log("Checkpoint: generated first fully connected layer");

        // Dropout
        const drop = MNIST_Model.dropout(full_1, 0.4);
        log("Checkpoint: performed dropout");


        // Map the 1024 features to 10 classes, one for each digit
        const weight_fc2 = MNIST_Model.weight_variable([1024, 10]);
        const bias_fc2 = MNIST_Model.bias_variable([10]);
        log("Checkpoint: mapped model to classifiers");

        // Finish training
        return dl.matMul(drop, weight_fc2) + bias_fc2;
        log("Checkpoint: finished training!");



    }

    /**
     * Generate the first convolutional layer from the inputted image data
     * @param x
     * @return {*}
     */
    static generate_first_conv_layer(x){
        let weight_conv1 = MNIST_Model.weight_variable([5, 5, 1, 32]);
        let bias_conv1 = MNIST_Model.bias_variable([32]);
        const conv_1 = dl.conv2d(x, weight_conv1, 1, 'same').add(bias_conv1);
        return conv_1.relu();
    }

    /**
     * Generate the second convolutional layer from the first pool layer
     * @param pool_1
     * @return {*}
     */
    static generate_second_conv_layer(pool_1){
        let weight_conv2 = MNIST_Model.weight_variable([5, 5, 32, 64]);
        let bias_conv2 = MNIST_Model.bias_variable([64]);
        const conv_2 = dl.conv2d(pool_1, weight_conv2, 1, 'same').add(bias_conv2);
        return conv_2.relu();
    }

    /**
     * Fully connects the second pool layer
     * @param pool_2
     * @return {*}
     */
    static generate_fully_connected_layer(pool_2){
        let weight_full1 = MNIST_Model.weight_variable([7 * 7 * 64, 1024]);
        let bias_full1 = MNIST_Model.bias_variable([1024]);

        const pool2_flat = dl.reshape(pool_2, [-1, 7 * 7 * 64]);
        const full = dl.matMul(pool2_flat,weight_full1).add(bias_full1);
        return full.relu()
    }

    static dropout(x, keep_prob) {

        if (keep_prob > 1 || keep_prob < 0) {
            throw "Keep probability must be between 0 and 1"
        }

        if (keep_prob === 1) return x;

        let random_tensor = dl.fill(x.shape, keep_prob).add(dl.randomUniform(x.shape));
        random_tensor = dl.floor(random_tensor);

        return x.div(dl.scalar(keep_prob)).mul(random_tensor)
    }

    /**
     * Generates a weight variable of a given shape
     * @param shape
     */
    static weight_variable(shape){
        let initial =  dl.truncatedNormal(shape);
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

}

let mnist_model = new MNIST_Model();








