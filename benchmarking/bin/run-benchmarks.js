#!/usr/local/bin/node

/**
 * CLI Configuration
 * =================================================
 */

/**
 * Constants
 */
const config = require('config');

const valid_environments = config.valid_environemnts;
const valid_backends = config.valid_backends;
const default_iterations = config.default_iterations;

const benchmark_dir = config.benchmark_dir;
const python_dir = config.python_dir;
const js_dir = config.js_dir;
const current_dir = __dirname;
const spawner_url = `${current_dir}/headless-browser-spawner.js`;
const output_dir = `${current_dir}/../output`;

const benchmark_mapping = {
    mnist: `${benchmark_dir}/MNIST/`,
    squeezenet: `${benchmark_dir}/SqueezeNet/`
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
const ArgumentParser = require('arg-parser').ArgParser;
let parser = new ArgumentParser('benchmarks');
parser = parser.get_arg_parser();


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

// Set backend settings
if(args.backend){
    if (args.backend === 'cpu' || args.backend === 'gpu'){
        backends = [args.backend]
    } else {
        throw_error(`Invalid verbose input: ${args.backend}`)
    }
} else {
    backends = valid_backends
}

// Set logging settings
if(args.verbose){
    if (args.verbose === 'true' || args.verbose === 'false'){
        verbose = (args.verbose === 'true')
    } else {
        throw_error(`Invalid verbose input: ${args.verbose}`)
    }
}

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

// Get platform
const platform = args.platform;
if(!platform) throw_error(`Platform not specified`);

/**
 * Print out configuration to user
 */
log(chalk.blue.bold('Benchmarks: ') + chalk.blue(tests_performing));
log(chalk.blue.bold('Environments: ') + chalk.blue(envs_running));
log(chalk.blue.bold('Backends: ') + chalk.blue(backends));
log(chalk.blue.bold('Iterations: ') + chalk.blue(iterations));
log(chalk.blue.bold('Verbose: ') + chalk.blue(verbose));


/**
 * Now that we have the inputs, time to run stuff!
 */
print_lines(1);

const total = new ProgressBar('  Total percentage completed [:bar] :percent :etas \n', {
            complete: '=',
            incomplete: ' ',
            width: 30,
            total: iterations * tests_performing.length * envs_running.length * backends.length
        });

tests_performing.forEach(benchmark => {
    log(chalk.blue.bgGreen(`Starting benchmark: ${benchmark}`));
    envs_running.forEach(environ => {

        prep_environment(environ, benchmark);

        log(chalk.blue.bgYellow(`Running in environment: ${environ}`));

        backends.forEach(backend => {

            log(chalk.bgMagenta(`Running with backend: ${backend}`));

            // The name of the file to have results outputted to
            const output_file = `${output_dir}/${benchmark}/${platform}-${environ}-${backend}-${Math.floor(Date.now() / 1000)}.txt`;

            for(let x = 0; x < iterations; x++) {
                run_benchmark(benchmark, environ, output_file, backend);
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
 * @param backend
 */
function run_benchmark(benchmark, env, output_file, backend){
    switch (env) {
        case 'python':
            run_benchmark_in_python(benchmark, backend, output_file);
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
 * @param backend
 * @param output_file
 */
function run_benchmark_in_python(benchmark, backend, output_file) {
    const runner = `python3 ${current_dir}/../${benchmark_mapping[benchmark]}/${python_dir}/runner.py --backend ${backend} --output ${output_file}`;
    run_cmd(runner)
}


/**
 * Calls bash command for running a headless browser implementation of benchmark
 * @param benchmark
 * @param browser
 * @param output_file
 * @param backend
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
    const builder = `cd ${current_dir}/../${benchmark_mapping[benchmark]}/${js_dir} && yarn test`;
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


