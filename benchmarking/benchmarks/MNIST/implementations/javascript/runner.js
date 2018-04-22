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

    let start_test = performance.now();
    log("start of test: ", start_test);

    await benchmark.predict(batch);

    let end_test = performance.now();

    log("end of test: ", end_test);

    let test_time = (end_test - start_test);

    console.log(JSON.stringify({
            options: `test( ${backend} )`,
            time: test_time,
        }));

    return {train: train_time, test: test_time}
}


(async ()=>{

    for(const x of [...Array(10).keys()]) {
        let result = await runner('gpu');
        log(result)
    }




    // let gpu_promises = [];
    // let cpu_promises = [];
    //
    //
    // for(let x = 0; x < 10; x++){
    //     gpu_promises.push(runner_gpu);
    //     cpu_promises.push(runner_cpu);
    // }
    //
    // log("gpu promises: ", gpu_promises);
    //
    // Promise.all(gpu_promises).then(times => {
    //     console.log("GPU TIMES");
    //     console.log("===================");
    //     console.log(times);
    // });
    //
    // Promise.all(cpu_promises).then(times => {
    //     console.log("CPU TIMES");
    //     console.log("===================");
    //     console.log(times);
    // });

})();
