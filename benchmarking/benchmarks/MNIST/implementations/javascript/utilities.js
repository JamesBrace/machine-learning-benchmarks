import * as tf from '@tensorflow/tfjs';

export async function warmupAndBenchmarkGPU(benchmark){
    // Warmup.
    const out = benchmark();
    await out.data();
    out.dispose();
    return (await tf.time(benchmark)).kernelMs;
}
