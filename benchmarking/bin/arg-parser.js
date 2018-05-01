const ArgumentParser = require('argparse').ArgumentParser;
const config = require('./config.json');

class ArgParser {

    constructor(type){
        this.parser = new ArgumentParser({
          version: '1.0.0',
          addHelp: true,
          description: 'Headless Browser Spawner'
        });

        this.construct_parser(type)
    }

    construct_parser(type){
        switch(type){
            case 'browser-spawner':
                this.create_browser_spawner_args();
                break;
            case 'benchmarks':
                this.create_benchmark_args()
        }
    }

    create_benchmark_args(){
        this.parser.addArgument([ '-t', '--test' ],
            {
                help: `The test you want to run. Current options are 'mnist', 'squeezenet', and 'utilities'. Defaults to all.`
            }
        );

        this.parser.addArgument([ '-e', '--env' ],
            {
                help: `The environment you want to run in. Current options are 'python', 'chrome', and 'firefox'. Defaults to all.`
            }
        );

        this.parser.addArgument([ '-i', '--iterations' ],
            {
                help: `The number of iterations you want to run. Defaults to ${config.default_iterations}`
            }
        );

        this.parser.addArgument([ '-l', '--verbose'],
            {
                help: `Set to true if you want to turn on verbose mode. Defaults to 'false'`
            }
        );

        this.parser.addArgument([ '-b', '--backend'],
            {
                help: `The backend you want to run the benchmarks on. Can either be 'cpu' or 'gpu'. Defaults to 'both'`
            }
        );

        this.parser.addArgument([ '-p', '--platform'],
            {
                help: `The platform you are running the benchmarks on.`
            }
        );
    }

    create_browser_spawner_args(){
        this.parser.addArgument([ '-t', '--test' ],
            {
                help: `The test you want to run. Current options are 'mnist', 'squeeznet', and 'utilities'`
            }
        );

        this.parser.addArgument([ '-b', '--browser' ],
            {
                help: `The browser you want to run the test in. Current options are 'firefox' and 'chrome'. Default is both.`
            }
        );

        this.parser.addArgument([ '-be', '--backend' ],
            {
                help: `The backend you want to run the test in. Current options are 'cpu' and 'gpu'. Default is both.`
            }
        );


        this.parser.addArgument([ '-o', '--output' ],
            {
                help: `The name of file you want to save output to.`
            }
        );
    }

    get_arg_parser(){
        return this.parser;
    }
}

module.exports = ArgParser;