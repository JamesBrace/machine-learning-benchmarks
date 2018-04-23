import * as mnist from './mnist';
import * as tf from '@tensorflow/tfjs';

/**
 * BROWSER CONFIGURATION
 */

let isChrome = !!window.chrome && !!window.chrome.webstore;
const browser = (isChrome) ? 'chrome' : 'firefox';
log(`Info: current browser is ${browser.toUpperCase()}`);

async function runner(backend) {

    console.log("Info: Setting up benchmark");

    let benchmark = await mnist.setup(backend);

    console.log("Info: Model created");

    let start;
    let end;

    console.log(`Info: Backend being used is '${tf.getBackend()}'`);

    if(backend === 'gpu') {
        // Warmup
        log("Info: Warming up gpu");
        await benchmark.train();
    }

    log("Info: Starting train benchmark");

    start = performance.now();
    await benchmark.train();
    end = performance.now();

    log("Info: Finished train benchmark");

    let train_time = end - start;

    console.log("Info: " + JSON.stringify({
        options: `train( ${backend} )`,
        time: train_time,
    }));

    let batch = mnist.nextBatch('test', 64);
    const iterations = 50;

    console.log("Info: Starting prediction testing with 50 iterations");

    let start_test = performance.now();

    for(const x of [...Array(iterations).keys()]) {
        await benchmark.predict(batch);
    }

    let end_test = performance.now();

    console.log("Info: Finished prediction testing");

    let test_time = (end_test - start_test)/iterations;

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
function log(text){
    if(isChrome){
        console.log(text)
    } else {
        const output = document.getElementById('output');
        output.innerText = text;
         // Used to stop browser spawner in FireFox
        document.title = "Done"
    }
}


(async ()=>{
    log("Info: Starting Benchmark");
    document.title = "Done";

    let result = await runner('gpu');
    log(JSON.stringify(result));

    console.log("Info: Finished Benchmark");

    // Used to stop browser spawner in Chrome
    window.name = "Close";

})();
