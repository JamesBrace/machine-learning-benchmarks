const fs = require('fs');
const ArgParser = require('./arg-parser');
const config = require('./config');
require('geckodriver');
const {Builder, By, until} = require('selenium-webdriver');
const path = require('path');
const firefox = require('selenium-webdriver/firefox');
const puppeteer = require('puppeteer');

/**
 * Arg Parser
 */
let parser = new ArgParser('browser-spawner');
parser = parser.get_arg_parser();
const args = parser.parseArgs();

let backend;
if (args.backend === 'gpu' || args.backend === 'cpu'){
    backend = args.backend;
} else {
    throw new Error(`Invalid backend: ${args.backend}`)
}

/**
 * Firefox Headless Setup
 */
const path_to_firefox = config.firefox_path;

// The global path to output file which is used by the program
const file_url = args.output;

/**
 * URL Mapping
 */
const path_to_js = config.js_path;
const urls = {
    mnist: `../benchmarks/MNIST/${path_to_js}/index-${backend}.html`,
    squeezenet: `http://localhost:1337/${backend}`,
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
            '--hide-scrollbars',
            '--mute-audio',
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
                timeout: 30000000000
            });
}

/**
 * Opens headless Firefox browser and waits for an alert to be present
 * @param url
 * @return {Promise<void>}
 */
async function load_and_capture_firefox(url){
    const options = new firefox.Options();
    options.setBinary(path_to_firefox);

    const driver = await new Builder()
        .forBrowser('firefox')
        .setFirefoxOptions(options)
        .build();

    await driver.get(url);

    let el = await driver.wait(until.alertIsPresent(), 1000000000000);

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

