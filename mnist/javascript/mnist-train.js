const log = console.log;

/*****************************
 *  CONSTANTS
 ****************************/

// Data set sizes
const TRAINING_SIZE = 8000;
const TEST_SIZE = 2000;
const set = mnist.set(TRAINING_SIZE, TEST_SIZE);

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
const LEARNING_RATE = .001;
const BATCH_SIZE = 64;
const TRAIN_STEPS = 1000;

// Data constants.
const IMAGE_SIZE = 28;
const LABELS_SIZE = 10;
const optimizer = dl.train.adam(LEARNING_RATE);

// Variables that we want to optimize
const conv1OutputDepth = 32;
const conv1Weights = dl.variable(dl.randomNormal([5, 5, 1, conv1OutputDepth], 0, 0.1));

const conv2InputDepth = conv1OutputDepth;
const conv2OutputDepth = 64;
const conv2Weights = dl.variable(dl.randomNormal([5, 5, conv2InputDepth, conv2OutputDepth], 0, 0.1));

const fullyConnectedWeights1 = dl.variable(dl.randomNormal([7 * 7 * conv2OutputDepth, 1024], 0, 0.1));
const fullyConnectedBias1 = dl.variable(dl.zeros([1024]));

const fullyConnectedWeights2 = dl.variable(
    dl.randomNormal([1024, LABELS_SIZE], 0, 0.1));
const fullyConnectedBias2 = dl.variable(dl.zeros([10]));

// Loss function
function loss(labels, logits) {
    return dl.losses.softmaxCrossEntropy(labels, logits).mean();
}

// Our actual model
function model(inputXs, training) {
    const xs = inputXs.as4D(-1, IMAGE_SIZE, IMAGE_SIZE, 1);

    const strides = 2;
    const keep_prob = 0.4;

    // Conv 1
    const layer1 = dl.tidy(() => {
        return xs.conv2d(conv1Weights, 1, 'same')
            .relu()
            .maxPool([2, 2], strides, 'same');
    });

    // Conv 2
    const layer2 = dl.tidy(() => {
        return layer1.conv2d(conv2Weights, 1, 'same')
            .relu()
            .maxPool([2, 2], strides, 'same');
    });


    // Dense layer
    const full = dl.tidy(() => {
        return layer2.as2D(-1, 7 * 7 * 64)
            .matMul(fullyConnectedWeights1)
            .add(fullyConnectedBias1)
            .relu();
    });

    if(training){
        // Dropout
        const dropout = dl.tidy(() => {
            if (keep_prob > 1 || keep_prob < 0) {
                throw "Keep probability must be between 0 and 1"
            }

            if (keep_prob === 1) return full;

            const uniform_tensor = dl.randomUniform(full.shape);
            const prob_tensor = dl.fill(full.shape, keep_prob);
            const random_tensor = dl.add(uniform_tensor, prob_tensor);
            const floor_tensor = dl.floor(random_tensor);

            return full.div(dl.scalar(keep_prob)).mul(floor_tensor)
        });

        return dl.matMul(dropout || full , fullyConnectedWeights2).add(fullyConnectedBias2);

    }

    return dl.matMul(full , fullyConnectedWeights2).add(fullyConnectedBias2);
}

function nextTrainBatch(){
    let mapped = data.training.images.map((img, index) => {
            return {img: img, label: data.training.labels[index]}
        });

    const shuffled = mapped.sort(() => .5 - Math.random());// shuffle
    return {images: dl.tensor(shuffled.map(obj => obj.img).slice(0, BATCH_SIZE)),
        labels: dl.tensor(shuffled.map(obj => obj.label).slice(0, BATCH_SIZE))}
}

// Train the model.
async function train() {
  const returnCost = true;

  for (let i = 0; i < TRAIN_STEPS; i++) {
    const cost = optimizer.minimize(() => {
      const batch = nextTrainBatch();
      return loss(batch.labels, model(batch.images, true));
    }, returnCost);

    log(`loss[${i}]: ${cost.dataSync()}`);

    await dl.nextFrame();
  }
}

// Predict the digit number from a batch of input images.
function predict(x){
  const pred = dl.tidy(() => {
    const axis = 1;
    return model(x, false).argMax(axis);
  });
  return Array.from(pred.dataSync());
}

// Given a logits or label vector, return the class indices.
function classesFromLabel(y) {
  const axis = 1;
  const pred = y.argMax(axis);

  return Array.from(pred.dataSync());
}

// async function test() {
//   const testExamples = 50;
//   const batch = nextTestBatch(testExamples);
//   const predictions = predict(batch.xs);
//   const labels = classesFromLabel(batch.labels);
// }

async function run_mnist() {
  await train();
  // test();
}

run_mnist();
