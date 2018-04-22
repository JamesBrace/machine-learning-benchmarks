// Look at david's paper to evaluate how many iterations one should
/**
 * Arg Parser
 */
const ArgumentParser = require('node_modules/argparse/lib/argparse');
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

const webdriver = require('selenium-webdriver');
const path = require('path');
const firefox = require('selenium-webdriver/firefox');

/**
 * Chrome Headless Setup
 */
const puppeteer = require('puppeteer');


/**
 * URL Mapping
 */
const urls = {
    mnist: "",
    squeezenet: "",
    utilities: ""
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
            await spawn_browser(test_url, b)
        }
    } else {
        await spawn_browser(test_url, browser)
    }
}


async function spawn_browser(url, browser){

}

async function load_and_capture_chrome(url){
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.goto(url, {waitUntil: 'networkidle2'});

    page.on('console', msg => {
        
      for (let i = 0; i < msg.args().length; ++i)
        console.log(`${i}: ${msg.args()[i]}`);
    });

    await browser.close();
}

async function load_and_capture_firefox(url){

}


async function capture(url) {
  const binary = new firefox.Binary(firefox.Channel.RELEASE);
  binary.addArguments('-headless'); // until newer webdriver ships

  const options = new firefox.Options();
  options.setBinary(binary);
  // options.headless(); once newer webdriver ships

  const driver = new Builder().forBrowser('firefox')
    .setFirefoxOptions(options).build();

  await driver.get(url);
  const data = await driver.takeScreenshot();
  fs.writeFileSync('./screenshot.png', data, 'base64');

  driver.quit();
}

capture('https://hacks.mozilla.org/');


