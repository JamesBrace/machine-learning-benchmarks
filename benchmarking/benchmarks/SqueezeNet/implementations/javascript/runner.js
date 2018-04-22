import {SqueezeNet} from "./squeezenet";
import * as squeeze from './squeezenet'

export async function runner (backend, mode){
    let model = new SqueezeNet();
    await squeeze.loadData();

    let start = performance.now();
    await model.train();
    let end = performance.now();

    if (mode === 'train'){
        console.log(JSON.stringify({
                status: 1,
                options: `train(${backend}, ${mode})`,
                time: (end - start) / 1000,
                output: 0
            }));

        return
    }

    let batch = model.nextBatch('test', 100);
    start = performance.now();
    for(let x = 0; x < 50; x++) {
        model.predict(batch.images);
    }
    end = performance.now();

    console.log(JSON.stringify({
                status: 1,
                options: `test(${size})`,
                time: (end - start) / 1000,
                output: 0
            }));

}