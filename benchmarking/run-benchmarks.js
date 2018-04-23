#!/usr/local/bin/node

/**
 * CLI Configuration
 * =================================================
 */

/**
 * Constants
 */

const valid_environments = ['python', 'chrome', 'firefox'];
const default_iterations = 20;

const python_cmd = 'python3';

const benchmark_dir = 'benchmarks';
const impl_dir = 'implementations';

const benchmark_mapping = {
    mnist: `./${benchmark_dir}/MNIST/`,
    squeezenet: `./${benchmark_dir}/SqueezeNet/`,
    utilities: `./${benchmark_dir}/Utility-ML-Functions/`,
};

/**
 * Chalk
 */
const chalk = require('chalk');
const log = console.log;

/**
 * Arg Parser
 */
const ArgumentParser = require('argparse').ArgumentParser;
console.log(ArgumentParser);
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
  [ '-v', '--verbose' ],
  {
    help: `Set to true if you want to turn on verbose mode. Defaults to 'false'`
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
let verbose = true;

/**
 * Validate input
 */
const args = parser.parseArgs();

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

tests_performing.forEach(benchmark => {
    envs_running.forEach(environ => {

    }
});


/**
 * CLI Utilities
 * =================================================
 */

/**
 *
 */
function run_benchmark(env, benchmark){
    switch (env) {
        case 'python':
            run_benchmark_in_python(benchmark);
            break;
        case 'chrome':
        case 'firefox':
            run_benchmark_in_browser(benchmark, env);
            break;
        default:
            throw_error(`Invalid benchmark and environment pairing. Could not run ${benchmark} in ${env}`)
    }
}


/**
 * Calls bash command for running a benchmark in python
 * @param benchmark
 */
function run_benchmark_in_python(benchmark){

}

/**
 * Calls bash command for running a headless browser implementation of benchmark
 * @param benchmark
 * @param browser
 */
function run_benchmark_in_browser(benchmark, browser){

}

/**
 * Throws error with cool red message + exits program
 * @param error
 */
function throw_error(error){
    log(chalk.red(error));
    process.exit(0)
}

/**
 * Controls whether benchmark logging is on or off
 */
const logger = function() {
    let oldConsoleLog = null;
    let pub = {};

    pub.enableLogger =  function enableLogger()
                        {
                            if(oldConsoleLog == null)
                                return;

                            window['console']['log'] = oldConsoleLog;
                        };

    pub.disableLogger = function disableLogger()
                        {
                            oldConsoleLog = console.log;
                            window['console']['log'] = function() {};
                        };

    return pub;
}();



