const { spawn } = require('child_process');

console.log("Starting Cifar10 server")

const subprocess = spawn('node', ['./cifar10-server.js'], {
  detached: true,
});

subprocess.unref();
