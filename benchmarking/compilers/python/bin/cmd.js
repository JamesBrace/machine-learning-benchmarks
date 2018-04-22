#!/usr/bin/env node
var fs = require('fs');
var path = require('path');
var argv = require('minimist')(process.argv.slice(2));
var MultiStream = require('multistream');

if (argv._[0] === 'help' || argv.h || argv.help
|| typeof argv.f === 'undefined'
|| (process.argv.length <= 2 && process.stdin.isTTY)) {
    return fs.createReadStream(__dirname + '/usage.txt')
        .pipe(process.stdout)
        .on('close', function () { process.exit(1) })
    ;
}

var inputs;
if (typeof argv.f === 'string') {
    inputs = [argv.f];
} else {
    inputs = argv.f;
}

var basedir = argv.b || (inputs[0] && path.dirname(inputs[0])) || process.cwd();
inputs.map(x => fs.createReadStream(x).pipe(fs.createWriteStream(basedir + x)));





