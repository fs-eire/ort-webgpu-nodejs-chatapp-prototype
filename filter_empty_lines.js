import fs from 'fs';
import readline from 'readline';
import { EOL } from 'os';

/**
 * Filters empty lines from input file and writes non-empty lines to output file
 * @param {string} inputPath - Path to the input file
 * @param {string} outputPath - Path to the output file
 */
async function filterEmptyLines(inputPath, outputPath) {
    try {
        // Create read and write streams
        const fileStream = fs.createReadStream(inputPath);
        const writeStream = fs.createWriteStream(outputPath);
        
        const rl = readline.createInterface({
            input: fileStream,
            crlfDelay: Infinity
        });
        
        let linesRead = 0;
        let linesWritten = 0;
        
        // Process each line
        for await (const line of rl) {
            linesRead++;
            
            // Skip empty lines or lines with only whitespace
            if (line.trim() === '') {
                continue;
            }
            
            // Write non-empty lines to output file
            writeStream.write(`${line}${EOL}`);
            linesWritten++;
        }
        
        writeStream.end();
        console.log(`Processing complete!`);
        console.log(`Lines read: ${linesRead}`);
        console.log(`Lines written: ${linesWritten}`);
        console.log(`Empty lines removed: ${linesRead - linesWritten}`);
    } catch (error) {
        console.error(`Error: ${error.message}`);
    }
}

// Example usage
const inputFile = process.argv[2] || 'node_logs.txt';
const outputFile = process.argv[3] || 'node_logs.filtered.txt';

console.log(`Filtering empty lines from "${inputFile}" to "${outputFile}"...`);
filterEmptyLines(inputFile, outputFile);