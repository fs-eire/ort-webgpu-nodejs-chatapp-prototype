import fs from 'fs';
import { EOL } from 'os';
import readline from 'readline';

async function processRawWebGpuData(inputFilePath, outputFilePath) {
    const fileStream = fs.createReadStream(inputFilePath);
    const writeStream = fs.createWriteStream(outputFilePath);

    const rl = readline.createInterface({
        input: fileStream,
        crlfDelay: Infinity
    });

    for await (const line of rl) {
        const result = /^\[.+INFO:CONSOLE\(\d+\)]\ "(?<json_data>.+)",\ source:\ [^"]+?\(\d+\)$/.exec(line);
        if (!result) {
            continue;
        }
        const jsonData = result.groups.json_data;
        writeStream.write(`${jsonData}${EOL}`);
    }

    writeStream.end();
    console.log('File writing completed.');
}

if (process.argv.length !== 4) {
    console.error('Usage: node pre-process_chrome_debug_log_webgpu_ep.js <inputFilePath> <outputFilePath>');
    process.exit(1);
}

// Call the function with the path to your input and output files
processRawWebGpuData(process.argv[2], process.argv[3])