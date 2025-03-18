import fs from 'fs';
import readline from 'readline';

/**
 * Process a file line by line and write results to output file
 * @param {string} inputFilePath - Path to the input file
 * @param {string} outputFilePath - Path to the output file
 */
async function processFileLineByLine(inputFilePath, outputFilePath) {
  try {
    // Create a readable stream for the input file
    const fileStream = fs.createReadStream(inputFilePath);
    
    // Handle potential errors with the file stream
    fileStream.on('error', (error) => {
      console.error(`Error reading file: ${error.message}`);
    });

    // Create interface for reading line by line
    const rl = readline.createInterface({
      input: fileStream,
      crlfDelay: Infinity
    });

    let output = '';

    // Process each line
    let line_head = 2;
    let lineNumber = 0;
    for await (const line of rl) {
      lineNumber++;

      output += line + '\n';

      if (line.includes(' wgpu')) {
        output += `printf("## ${line}\\n");\n`;
      }
    }

    console.log('File processing complete');
    fs.writeFileSync(outputFilePath, output);
    console.log(`Output written to: ${outputFilePath}`);
  } catch (error) {
    console.error(`Error: ${error.message}`);
  }
}

// Example usage
const inputFilePath = process.argv[2]; // D:\code\onnxruntime\build\Windows\RelWithDebInfo\_deps\dawn-build\gen\src\dawn\native\webgpu_dawn_native_proc.cpp
const outputFilePath = process.argv[3];
processFileLineByLine(inputFilePath, outputFilePath);