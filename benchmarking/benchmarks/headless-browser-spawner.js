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

const wd = require('selenium-webdriver');
const path = require('path');
const firefox = require('selenium-webdriver/firefox');

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
    squeezenet: "dasd",
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
    const watchDog = page.waitForFunction('window.name === "Close"', {timeout: 0});

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

async function load_and_capture_firefox(url){
    const options = new firefox.Options();
    const prefs = new wd.logging.Preferences();
    let driver;

    prefs.setLevel(wd.logging.Type.BROWSER, wd.logging.Level.ALL);
    options.setLoggingPrefs(prefs);

    const binary = new firefox.Binary(firefox.Channel.RELEASE);
    binary.addArguments('-headless');

    options.setBinary(binary);

    driver = new wd.Builder()
        .forBrowser('firefox')
        .setFirefoxOptions(options)
        .build();

    driver
        .get(`file://${path.resolve(__dirname, url)}`)
        .then(() => driver.manage().logs().get(wd.logging.Type.BROWSER))
        .then((logs) => {
            fs.appendFile("./output/firefox-output.txt", logs, err => {
                if(err) return console.log(err);
            });
        })
        .then(() => driver.quit());
}

////////////////////////////////////////////////////////////////////

run_test()
    .catch(err => {
        console.log(err)
    });

