import {SqueezeNet} from "./squeezenet";
import * as squeeze from './squeezenet'
import * as tf from '@tensorflow/tfjs';

/**
 * BROWSER CONFIGURATION
 */
let isFirefox = false;

export async function runner (backend){
    console.log("Info: Setting up benchmark");

    const model = squeeze.init(backend);

    console.log("Info: Model created");
    console.log(`Info: Backend being used is '${tf.getBackend()}'`);

    if(backend === 'gpu') {
        // Warmup
        console.log("Info: Warming up gpu");
        await model.train();
    }

    console.log("Info: Starting train benchmark");

    let start = performance.now();
    await model.train();
    let end = performance.now();

    console.log("Info: Finished train benchmark");

    let train_time = (end - start);

    console.log("Info: " + JSON.stringify({
        options: `train( ${backend} )`,
        time: train_time,
    }));

    let batch_size = 100;
    let batch = model.nextBatch('test', batch_size);
    const iterations = 50;

    console.log("Info: Starting prediction testing with 50 iterations");

    let start_test = performance.now();

    for(const x of [...Array(iterations).keys()]) {
        await model.predict(batch, batch_size);
    }

    let end_test = performance.now();
    console.log("Info: Finished prediction testing");

    let test_time = (end_test - start_test)/iterations;

    console.log("Info: " + JSON.stringify({
            options: `test( ${backend} )`,
            time: test_time,
        }));

    return {benchmark: 'SqueezeNet', backend: backend, implementation: 'JavaScript', train: train_time, test: test_time}

}

/**
 * Need to do work around for Firefox since it currently doesn't support Seleniums logging API. What I do here
 * is render the output into a window alert and then change the title of the page to signify to Selenium that the task has
 * been completed
 * @param text
 */
function log_output(text){

    if(!isFirefox){
        console.log(text);
    } else {
        alert(text);
    }

    // Trigger chrome to close
    document.title = 'Close'
}

(async ()=>{

    console.log("Info: Started Benchmark");

    isFirefox = navigator.userAgent.toLowerCase().indexOf('firefox') > -1;
    const browser = (isFirefox) ? 'firefox' :'chrome';

    console.log(`Info: Current browser is ${browser.toUpperCase()}`);

    console.log("Info: Starting Benchmark");

    let backend = document.title;
    let result = await runner(backend);

    console.log("Info: Finished Benchmark");

    log_output(JSON.stringify(result));
})();