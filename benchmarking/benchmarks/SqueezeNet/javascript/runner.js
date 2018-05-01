import {SqueezeNet} from "./squeezenet";
import * as squeeze from './squeezenet'
import * as tf from '@tensorflow/tfjs';

/**
 * BROWSER CONFIGURATION
 */
let isFirefox = false;

/**
 * Constants
 */
const TRAIN_SIZE = 1000;
const TEST_SIZE = 1000;
const TRAIN_EPOCHS = 1;
const WARMUP_EPOCHS = 1;

export async function runner (backend){
    console.log("Info: Setting up model and loading data from server");

    const model = await squeeze.init(backend, TRAIN_SIZE, TEST_SIZE);

    console.log("Info: Model created");
    console.log(`Info: Backend being used is '${tf.getBackend()}'`);

    let train_set =  model.nextBatch('train', TRAIN_SIZE);

    if(backend === 'gpu') {
        // Warmup
        console.log("Info: Warming up gpu");
        await model.train(train_set, TRAIN_SIZE, WARMUP_EPOCHS);
    }

    console.log("Info: Starting train benchmark");

    let start = performance.now();
    await model.train(train_set, TRAIN_SIZE, TRAIN_EPOCHS);
    let end = performance.now();

    // Clear training set from memory!!
    train_set = [];

    console.log("Info: Finished train benchmark");

    let train_time = end - start;

    console.log("Info: " + JSON.stringify({
        options: `train( ${backend} )`,
        time: train_time,
    }));

    let test_set = model.nextBatch('test', TEST_SIZE);

    console.log("Info: Starting prediction testing");

    let start_test = performance.now();

    await model.predict(test_set, TEST_SIZE);

    let end_test = performance.now();
    console.log("Info: Finished prediction testing");

    let test_time = end_test - start_test;

    console.log("Info: " + JSON.stringify({
            options: `test( ${backend} )`,
            time: test_time,
        }));

    return {benchmark: 'SqueezeNet', backend: backend, implementation: 'JavaScript', train: train_time, test: test_time,
        train_size: TRAIN_SIZE, training_steps: TRAIN_EPOCHS, test_size: TEST_SIZE}

}

/**
 * Need to do work around for Firefox since it currently doesn't support Seleniums logging API. What I do here
 * is render the output into a window alert and then change the title of the page to signify to Selenium that the task has
 * been completed
 * @param output
 */
function log_output(output){

    if(!isFirefox){
        console.log(`Info: Logging to file  ${output}`);
        console.log(output);
        console.log("Info: Done");
    } else {
        alert(output);
    }
}

(async ()=>{

    console.log("Info: Started Benchmark");

    isFirefox = navigator.userAgent.toLowerCase().indexOf('firefox') > -1;
    const browser = (isFirefox) ? 'firefox' :'chrome';

    console.log(`Info: Current browser is ${browser.toUpperCase()}`);

    console.log("Info: Starting Benchmark");

    // Get the desired backend from the document title
    let backend = document.title.toLowerCase();
    let result = await runner(backend);

    console.log("Info: Finished Benchmark");

    log_output(JSON.stringify(result));
})();