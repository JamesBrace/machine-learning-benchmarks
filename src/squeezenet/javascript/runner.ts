import {SqueezeNet} from "./squeezenet";

function runner (size){
    let squeeze = new SqueezeNet();

    let start = performance.now();
    squeeze.run_squeeze()
        .then(res => {
            let end = performance.now();

            console.log(JSON.stringify({
                status: 1,
                options: `run (${size})`,
                time: (end - start) / 1000,
                output: 0
            }))
        })
}