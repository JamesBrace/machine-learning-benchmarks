// Look at david's paper to evaluate how many iterations one should
const fs = require('fs');

/**
 * Arg Parser
 */
const ArgumentParser = require('argparse').ArgumentParser;
console.log(ArgumentParser);
const parser = new ArgumentParser({
  version: '0.0.1',
  addHelp:true,
  description: 'Headless Browser Spawner'
});

parser.addArgument(
  [ '-t', '--test' ],
  {
    help: `The test you want to run. Current options are 'mnist', 'squeeznet', and 'utilities'`
  }
);

parser.addArgument(
  [ '-b', '--browser' ],
  {
    help: `The browser you want to run the test in. Current options are 'firefox' and 'chrome'. Default is both.`
  }
);

parser.addArgument(
  [ '-be', '--backend' ],
  {
    help: `The backend you want to run the test in. Current options are 'cpu' and 'gpu'. Default is both.`
  }
);


parser.addArgument(
  [ '-o', '--output' ],
  {
    help: `The name of file you want to save output to.`
  }
);

const args = parser.parseArgs();

// The relative location of the output file
// const file = `../output/${args.test}/${args.output}`;

let backend;
if (args.backend === 'gpu' || args.backend === 'cpu'){
    backend = args.backend;
} else {
    throw new Error(`Invalid backend: ${args.backend}`)
}

/**
 * Firefox Headless Setup
 */
require('geckodriver');

const {Builder, By, until} = require('selenium-webdriver');
const path = require('path');
const firefox = require('selenium-webdriver/firefox');
const path_to_firefox = "/jet/prs/workspace/firefox/firefox-bin";

// The global path to output file which is used by the program
const file_url = args.output;

/**
 * Chrome Headless Setup
 */
const puppeteer = require('puppeteer');


/**
 * URL Mapping
 */
const path_to_js = 'implementations/javascript';
const urls = {
    mnist: `../benchmarks/MNIST/${path_to_js}/index-${backend}.html`,
    squeezenet: `http://localhost:1337/${backend}`,
    utilities: "sadasd"
};

/**
 * Valid Browsers
 * @type {string[]}
 */

const browsers = {
    'chrome': load_and_capture_chrome,
    'firefox': load_and_capture_firefox
};

///////////////////////////////////////////////////////////////////////

async function run_test(){
    const test_url = urls[args.test];
    if(!test_url) throw new Error(`Invalid test option: ${args.test}`);

    // Handles the case where the url is a local file vs localhost
    const spawning_url = (test_url.includes('localhost'))? test_url : `file://${path.resolve(__dirname, test_url)}`;

    let browser = (args.browser && Object.keys(browsers).includes(args.browser)) ? args.browser : 'all';

    if (browser === 'all') {
        for (b of browsers){
            await browsers[b](spawning_url)
        }
    } else {
        await browsers[browser](spawning_url);
    }
}

async function load_and_capture_chrome(url){
    const browser = await puppeteer.launch({
        headless: false,
        args: [
            '--headless',
            '--hide-scrollbars',
            '--mute-audio',
            '--ignoreDefaultArgs',
            '--dumpio'
            // '--no-sandbox'
        ]
    });
    const page = await browser.newPage();

    page.on('console', msg => {
        msg.args().forEach(arg => {
            if(msg.text().includes("Info")) {
                console.log(msg.text());

                if(msg.text().includes("Done")){
                    browser.close()
                        .then(() => {
                            console.log("Exiting spawner successfully");
                            process.exit(0)
                        })
                        .catch((err) => {
                            console.log(`Exiting spawner with error: ${err}`);
                            process.exit(0)
                        })
                }

            } else {
                const output = msg.text();
                fs.appendFile(file_url, `${output} \n`, err => {
                    if(err) return console.log(err);
                });
            }
        });
    });

    await page.goto(url, {
                timeout: 3000000
            });
}

/**
 * Opens headless Firefox browser and waits for an alert to be present
 * @param url
 * @return {Promise<void>}
 */
async function load_and_capture_firefox(url){
    const options = new firefox.Options();
    options.headless();
    options.setBinary(path_to_firefox);

    const driver = await new Builder()
        .forBrowser('firefox')
        .setFirefoxOptions(options)
        .build();

    await driver.get(url);

    let el = await driver.wait(until.alertIsPresent(), 1000000000);

    const output = await el.getText();

    fs.appendFile(file_url, `${output} \n`, async err => {
        if(err) return console.log(err);
        await driver.quit()
    });
}

////////////////////////////////////////////////////////////////////

run_test()
    .catch(err => {
        console.log(err)
    });

