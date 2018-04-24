import * as mnist from './mnist';
import * as tf from '@tensorflow/tfjs';

/**
 * BROWSER CONFIGURATION
 */
let isFirefox = false;

/**
 * CONSTANTS
 */
const TRAIN_STEPS = 10;
const WARMUP_STEPS = 1;

async function runner(backend) {

    console.log("Info: Setting up benchmark");

    let benchmark = await mnist.init(backend);

    console.log("Info: Model created");

    let start;
    let end;

    console.log(`Info: Backend being used is '${tf.getBackend()}'`);

    let train_batch = mnist.nextBatch('train');

    if(backend === 'gpu') {
        // Warmup
        console.log(`Info: Warming up gpu with ${WARMUP_STEPS} iterations`);
        await benchmark.train(train_batch, WARMUP_STEPS);
    }

    console.log(`Info: Starting train benchmark with ${TRAIN_STEPS} epochs`);

    start = performance.now();
    await benchmark.train(train_batch, TRAIN_STEPS);
    end = performance.now();

    // Clear memory!!
    train_batch = [];

    console.log("Info: Finished train benchmark");

    let train_time = end - start / TRAIN_STEPS * 1000;

    console.log("Info: " + JSON.stringify({
        options: `train( ${backend} )`,
        time: train_time,
    }));

    let batch = mnist.nextBatch('test');

    console.log(`Info: Starting prediction testing`);

    let start_test = performance.now();
    await benchmark.predict(batch);
    let end_test = performance.now();

    console.log("Info: Finished prediction testing");

    let test_time = end_test - start_test;

    console.log("Info: " + JSON.stringify({
            options: `test( ${backend} )`,
            time: test_time,
        }));

    return {benchmark: 'MNIST', backend: backend, implementation: 'JavaScript', train: train_time, test: test_time}
}

/**
 * Need to do work around for Firefox since it currently doesn't support Seleniums logging API. What I do here
 * is render the output into a div and then change the title of the page to signify to Selenium that the task has
 * been completed
 * @param text
 */
function log_output(text){

    if(!isFirefox){
        console.log(`Info: Logging to file  ${text}`);
        console.log(text);
        console.log("Info: Done");
    } else {
        alert(text);
    }
}


(async ()=>{

    isFirefox = navigator.userAgent.toLowerCase().indexOf('firefox') > -1;
    const browser = (isFirefox) ? 'firefox' :'chrome';

    console.log(`Info: current browser is ${browser.toUpperCase()}`);

    console.log("Info: Starting Benchmark");

    // Get the desired backend from the document title
    let backend = document.title.toLowerCase();
    let result = await runner(backend);

    console.log("Info: Finished Benchmark");

    log_output(JSON.stringify(result));
})();
