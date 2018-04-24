#!/usr/local/bin/node

/**
 * CLI Configuration
 * =================================================
 */

/**
 * Constants
 */

const valid_environments = ['python', 'chrome', 'firefox'];
const valid_backends = ['gpu', 'cpu'];
const default_iterations = 20;


const benchmark_dir = 'benchmarks';
const impl_dir = 'implementations';
const python_dir = 'python';
const js_dir = 'javascript';
const current_dir = __dirname;
const spawner_url = `${current_dir}/headless-browser-spawner.js`;

const benchmark_mapping = {
    mnist: `${benchmark_dir}/MNIST/`,
    squeezenet: `${benchmark_dir}/SqueezeNet/`,
    utilities: `${benchmark_dir}/Utility-ML-Functions/`,
};

/**
 * Command Line Interfacing
 */
const execSync = require('child_process').execSync;


/**
 * Chalk
 */
const chalk = require('chalk');
const log = console.log;

/**
 * Progress Bar
 */
const ProgressBar = require('progress');


/**
 * Arg Parser
 */
const ArgumentParser = require('argparse').ArgumentParser;
const parser = new ArgumentParser({
  version: '1.0.0',
  addHelp:true,
  description: 'Benchmark spawner'
});

parser.addArgument(
  [ '-t', '--test' ],
  {
    help: `The test you want to run. Current options are 'mnist', 'squeezenet', and 'utilities'. Defaults to all.`
  }
);

parser.addArgument(
  [ '-e', '--env' ],
  {
    help: `The environment you want to run in. Current options are 'python', 'chrome', and 'firefox'. Defaults to all.`
  }
);

parser.addArgument(
  [ '-i', '--iterations' ],
  {
    help: `The number of iterations you want to run. Defaults to ${default_iterations}`
  }
);

parser.addArgument(
  [ '-l', '--verbose'],
  {
    help: `Set to true if you want to turn on verbose mode. Defaults to 'false'`
  }
);

parser.addArgument(
  [ '-b', '--backend'],
  {
    help: `The backend you want to run the benchmarks on. Can either be 'cpu' or 'gpu'. Defaults to 'both'`
  }
);

/**
 * CLI Logic
 * =================================================
 */

// The specs that are going to be ran
let tests_performing = [];
let envs_running = [];
let iterations = [];
let verbose = false;
let backends = '';

/**
 * Validate input
 */
const args = parser.parseArgs();

// Set logging settings
if(args.backend){
    if (args.backend === 'cpu' || args.backend === 'gpu'){
        backends = [args.backend]
    } else {
        throw_error(`Invalid verbose input: ${args.backend}`)
    }
} else {
    backends = valid_backends
}

log(chalk.blue.bold('Backend: ') + chalk.blue(backends));

// Set logging settings
if(args.verbose){
    if (args.verbose === 'true' || args.verbose === 'false'){
        verbose = (args.verbose === 'true')
    } else {
        throw_error(`Invalid verbose input: ${args.verbose}`)
    }
}

log(chalk.blue.bold('Verbose: ') + chalk.blue(verbose));

// Get tests
const tests = args.test;

if(tests){
    if(Object.keys(benchmark_mapping).includes(tests)){
        tests_performing = [tests];
    } else {
        throw_error(`Invalid test input: ${tests}`)
    }
} else {
    tests_performing = Object.keys(benchmark_mapping);
}

log(chalk.blue.bold('Benchmarks: ') + chalk.blue(tests_performing));

// Get environments
const envs = args.env;
if(envs){
    if(valid_environments.includes(envs)){
        envs_running = [envs]
    } else {
        throw_error(`Invalid environment input: ${tests}`)
    }
} else {
    envs_running = valid_environments
}

log(chalk.blue.bold('Environments: ') + chalk.blue(envs_running));

// Get iterations
const iters = args.iterations;

if(iters) {
    if(parseInt(iters)){
        iterations = parseInt(iters)
    } else {
        throw_error(`Invalid iteration input: ${tests}`)
    }
} else {
    iterations = default_iterations
}

log(chalk.blue.bold('Iterations: ') + chalk.blue(iterations));

/**
 * Now that we have the inputs, time to run stuff!
 */

print_lines(1);

const total = new ProgressBar('  Total percentage completed [:bar] :percent :etas \n', {
            complete: '=',
            incomplete: ' ',
            width: 50,
            total: iterations * tests_performing.length * envs_running.length * backends.length
        });

tests_performing.forEach(benchmark => {
    log(chalk.blue.bgGreen(`Starting benchmark: ${benchmark}`));
    envs_running.forEach(environ => {

        prep_environment(environ, benchmark);

        log(chalk.blue.bgYellow(`Running in environment: ${environ}`));

        const bar = new ProgressBar(`  ${environ} benchmarks completed [:bar] :current/${iterations} :percent :etas \n`, {
            complete: '=',
            incomplete: ' ',
            width: 20,
            total: iterations
        });

        bar.render();

        backends.forEach(backend => {

            log(chalk.bgMagenta(`Running with backend: ${backend}`));

            // The name of the file to have results outputted to
            const output_file = `${environ}-${backend}-${Math.floor(Date.now() / 1000)}.txt`;

            for(let x = 0; x < iterations; x++) {
                run_benchmark(benchmark, environ, output_file, backend);
                bar.tick();
                bar.render();
                total.tick();
                total.render()
            }
        });


        log(chalk.green(`Benchmarking in ${environ} completed.`))
    });
});


/**
 * CLI Utilities
 * =================================================
 */


/**
 * Runs the benchmark in the specified environment
 * @param env
 * @param benchmark
 * @param output_file
 */
function run_benchmark(benchmark, env, output_file, backend){
    switch (env) {
        case 'python':
            run_benchmark_in_python(benchmark, output_file, backend);
            break;
        case 'chrome':
        case 'firefox':
            run_benchmark_in_browser(benchmark, env, output_file, backend);
            break;
        default:
            throw_error(`Invalid benchmark and environment pairing. Could not run ${benchmark} in ${env}`)
    }
}


/**
 * Calls bash command for running a benchmark in python
 * @param benchmark
 */
function run_benchmark_in_python(benchmark) {
    const runner = `python ${current_dir}/../${benchmark_mapping[benchmark]}/${impl_dir}/${python_dir}/runner.py`;
    run_cmd(runner)
}


/**
 * Calls bash command for running a headless browser implementation of benchmark
 * @param benchmark
 * @param browser
 * @param output_file
 */
function run_benchmark_in_browser(benchmark, browser, output_file, backend){
    const runner = `node ${spawner_url} -t ${benchmark} -b ${browser} -o ${output_file} -be ${backend}`;
    run_cmd(runner)
}

/**
 * Builds javascript for browser environment
 * @param env
 * @param benchmark
 */
function prep_environment(env, benchmark){
    if(env === 'python') return;
    log(chalk.blue("Prepping environment..."));

    const current_dir = __dirname;
    const builder = `cd ${current_dir}/../${benchmark_mapping[benchmark]}/${impl_dir}/${js_dir} && yarn test`;
    run_cmd(builder);

    log(chalk.blue("Finish prepping environment"));
}

/**
 * Runs synchronous command in shell
 * @param cmd
 */
function run_cmd(cmd) {

    const out = (verbose) ? {stdio:[0,1,2]} : null;

    try {
       execSync(cmd, out)
    } catch (ex) {
        throw_error(ex.stdout);
    }
}

/**
 * Throws error with cool red message + exits program
 * @param error
 */
function throw_error(error){
    log(chalk.red(error));
    process.exit(0)
}

function print_lines(num_lines){
    for(let i = 0; i < num_lines; i++){
        log("")
    }
}


