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
                console.log("Computation time: " + (end - start) / 1000 + " s\n");
                console.log(JSON.stringify({
                    status: 1,
                    options: `train( ${backend} )`,
                    time: (end - start) / 1000,
                    output: 0
                }));
                return
            }

            start = performance.now();
            benchmark.predict(100);
            end = performance.now();

            console.log("Computation time: " + (end - start) / 1000 + " s\n");
            console.log(JSON.stringify({
                    status: 1,
                    options: `test( ${backend} )`,
                    time: (end - start) / 1000,
                    output: 0
                }));
        });
}