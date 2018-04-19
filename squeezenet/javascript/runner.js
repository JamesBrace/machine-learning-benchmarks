"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var squeezenet_1 = require("./squeezenet");
function runner(size) {
    var squeeze = new squeezenet_1.SqueezeNet();
    var start = performance.now();
    squeeze.run_squeeze()
        .then(function (res) {
        var end = performance.now();
        console.log(JSON.stringify({
            status: 1,
            options: "run (" + size + ")",
            time: (end - start) / 1000,
            output: 0
        }));
    });
}
