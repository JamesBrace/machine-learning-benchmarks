import * as mnist from './mnist';
import * as tf from '@tensorflow/tfjs';

const log = console.log;
async function runner(backend) {

    let benchmark = await mnist.setup(backend);

    log("model: ", benchmark);

    let start;
    let end;

    log(tf.getBackend());

    if(backend === 'gpu') {
        // Warmup
        log("warming up gpu");
        await benchmark.train();
    }

    start = performance.now();
    log("start  of train: ", start);
    await benchmark.train();
    end = performance.now();

    log("end  of train: ", end);

    let train_time = end - start;

    console.log(JSON.stringify({
        options: `train( ${backend} )`,
        time: train_time,
    }));


    let batch = mnist.nextBatch('test', 64);

    const iterations = 50;

    start = performance.now();
    log("start of test: ", start);

    for(let x = 0; x < iterations; x++){
        benchmark.predict(batch);
    }

    end = performance.now();

    log("end of test: ", start);

    let test_time = end - start / iterations;

    console.log(JSON.stringify({
            options: `test( ${backend} )`,
            time: test_time,
        }));

    return {train: train_time, test: test_time}
}

function runner_gpu(){
   return new Promise(resolve => {
       runner('gpu')
           .then(res => resolve(res))
   })
}

function runner_cpu(){
    return new Promise(resolve => {
       runner('cpu')
           .then(res => resolve(res))
   })
}

(async ()=>{
    let gpu_promises = [];
    let cpu_promises = [];


    for(let x = 0; x < 10; x++){
        gpu_promises.push(runner_gpu);
        cpu_promises.push(runner_cpu);
    }

    log("gpu promises: ", gpu_promises);

    Promise.all(gpu_promises).then(times => {
        console.log("GPU TIMES");
        console.log("===================");
        console.log(times);
    });

    Promise.all(cpu_promises).then(times => {
        console.log("CPU TIMES");
        console.log("===================");
        console.log(times);
    });

})();
