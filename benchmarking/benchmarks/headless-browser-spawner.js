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

const args = parser.parseArgs();

/**
 * Firefox Headless Setup
 */
require('geckodriver');

const {Builder, By, until} = require('selenium-webdriver');
const path = require('path');
const firefox = require('selenium-webdriver/firefox');
const path_to_firefox = "/Applications/Firefox.app/Contents/MacOS/firefox-bin";

/**
 * Chrome Headless Setup
 */
const puppeteer = require('puppeteer');


/**
 * URL Mapping
 */
const path_to_js = 'implementations/javascript';
const urls = {
    mnist: `MNIST/${path_to_js}/index.html`,
    squeezenet: `localhost:1337`,
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

    let browser = (args.browser && Object.keys(browsers).includes(args.browser)) ? args.browser : 'all';

    if (browser === 'all') {
        for (b of browsers){
            await browsers[b](test_url)
        }
    } else {
        await browsers[browser](test_url);
        console.log("done");

    }
}

async function load_and_capture_chrome(url){
    const browser = await puppeteer.launch({
        headless: false,
        args: [
          '--headless',
          '--hide-scrollbars',
          '--mute-audio'
        ]
    });
    const page = await browser.newPage();
    const watchDog = page.waitForFunction('document.title === "Close"', {timeout: 0});

    page.on('console', msg => {
        msg.args().forEach(arg => {
            if(msg.text().includes("Info")) {
                console.log(msg.text());
            } else {
                const output = msg.text();
                fs.appendFile("./output/MNIST/chrome-output.txt", `${output} \n`, err => {
                    if(err) return console.log(err);
                });
            }
        });
    });

    await page.goto(`file://${path.resolve(__dirname, url)}`);
    await watchDog;
    await browser.close();
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

    await driver.get(`file://${path.resolve(__dirname, url)}`);

    let el = await driver.wait(until.alertIsPresent(), 1000000000);

    const output = await el.getText();

    console.log("Output: ", output);

    await fs.appendFile("./output/firefox-output.txt", output, err => {
            if(err) return console.log(err);
    });

    await driver.quit()
}

////////////////////////////////////////////////////////////////////

run_test()
    .catch(err => {
        console.log(err)
    });

