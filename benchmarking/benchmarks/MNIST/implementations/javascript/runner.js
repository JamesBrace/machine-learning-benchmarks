import * as mnist from './mnist';

function runner(params) {

    const backend = params['backend'];
    const mode = params['mode'];

    let benchmark = mnist.setup(backend);

    let start = performance.now();
    let end;
    benchmark.train()
        .then(()=> {
            end = performance.now();

            if(mode === 'train'){
                console.log(JSON.stringify({
                    status: 1,
                    options: `train( ${backend} )`,
                    time: (end - start) / 1000,
                    output: 0
                }));
                return
            }

            let batch = mnist.nextBatch('test', 100);
            start = performance.now();

            for(let x = 0; x < 50; x++){
                benchmark.predict();
            }
            end = performance.now();

            console.log(JSON.stringify({
                    status: 1,
                    options: `test( ${backend} )`,
                    time: (end - start) / 1000,
                    output: 0
                }));
        });
}