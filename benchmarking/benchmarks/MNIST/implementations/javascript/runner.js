import * as mnist from './mnist';
import * as tf from '@tensorflow/tfjs';

const log = console.log;
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


(async ()=>{
    console.log("Info: Starting Benchmark");

    for(const x of [...Array(10).keys()]) {
        let result = await runner('gpu');
        console.log(JSON.stringify(result))
    }

    console.log("Info: Finished Benchmark");

    // Used to stop browser spawner
    window.name = "Close"


})();
