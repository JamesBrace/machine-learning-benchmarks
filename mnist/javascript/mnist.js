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

        this.train();
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

    train(){

        // Input layer
        let x = dl.tensor2d(this.data.training.images);
        x = x.reshape([-1, 28, 28, 1]);
        log("Checkpoint: generated input layer");

        // Convolution layer #1
        let filter_1 = dl.randomNormal([5, 5, 1, 32]);

        const conv_1 = dl.conv2d(x, filter_1, 1, 'same');
        const conv_1_post_activation = conv_1.relu();
        log("Checkpoint: generated first convolutional layer");

        // Pool layer #1
        const pool_1 = dl.maxPool(conv_1_post_activation, [2, 2], 2, 'valid');
        log("Checkpoint: generated first max pool layer");

        // Convolution layer #2
        let filter_2 = dl.randomNormal([5, 5, 32, 64]);

        const conv_2 = dl.conv2d(pool_1, filter_2, 2, 'same');
        const conv_2_post_activation = conv_2.relu();

        log("Checkpoint: generated second convolutional layer");

        // Pool layer #2
        const pool_2 = dl.maxPool(conv_2_post_activation, [2, 2], 2, 'valid');
        log("Checkpoint: generated second max pool layer");

        // Dense layer
        // const pool2_flat = dl.reshape(pool_2, [-1, 7 * 7 * 64]);

        const graph = new Graph();

    }

    createFullyConnectedLayer(graph, inputLayer, layerIndex,
        sizeOfThisLayer, includeRelu = true, includeBias = true) {
      return graph.layers.dense(
          'fully_connected_' + layerIndex, inputLayer, sizeOfThisLayer,
          includeRelu ? (x) => graph.relu(x) : undefined, includeBias);
    }
}

let mnist_model = new MNIST_Model();








