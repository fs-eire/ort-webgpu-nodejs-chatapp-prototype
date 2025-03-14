// Function to read and process the log file
function processLogFile(fileContent) {
    // Arrays to store the time differences
    const bMinusA = [];
    const xMinusW = [];

    // Variables to store the most recent timestamps for types 'a' and 'w'
    let lastATimestamp = null;
    let lastWTimestamp = null;

    // Split the content by newlines and process each line
    const lines = fileContent.trim().split('\n');

    for (const line of lines) {
        // Skip empty lines
        if (!line.trim()) continue;

        // Parse the line to get type and timestamp
        const [type, timestampStr] = line.trim().split(/\s+/);
        const timestamp = parseInt(timestampStr);

        if (isNaN(timestamp)) {
            console.error(`Invalid timestamp in line: ${line}`);
            continue;
        }

        // Process based on the type
        switch (type) {
            case 'a':
                lastATimestamp = timestamp;
                break;
            case 'b':
                if (lastATimestamp !== null) {
                    bMinusA.push(timestamp - lastATimestamp);
                    lastATimestamp = null; // Reset after finding a match
                }
                break;
            case 'w':
                lastWTimestamp = timestamp;
                break;
            case 'x':
                if (lastWTimestamp !== null) {
                    xMinusW.push(timestamp - lastWTimestamp);
                    lastWTimestamp = null; // Reset after finding a match
                }
                break;
        }
    }

    return { bMinusA, xMinusW };
}

// Function to calculate the percentile of an array
function calculatePercentile(arr, percentile) {
    if (arr.length === 0) return null;

    // Sort the array
    const sortedArr = [...arr].sort((a, b) => a - b);

    // Calculate the index
    const index = (percentile / 100) * (sortedArr.length - 1);

    // If the index is an integer, return the value at that index
    if (Number.isInteger(index)) {
        return sortedArr[index];
    }

    // Otherwise, interpolate between the two surrounding values
    const lowerIndex = Math.floor(index);
    const upperIndex = Math.ceil(index);
    const lowerValue = sortedArr[lowerIndex];
    const upperValue = sortedArr[upperIndex];
    const fraction = index - lowerIndex;

    return lowerValue + (upperValue - lowerValue) * fraction;
}

// Main function to process the file and output the results
function main(fileContent) {
    const { bMinusA, xMinusW } = processLogFile(fileContent);

    const medianBA = calculatePercentile(bMinusA, 50);
    const medianXW = calculatePercentile(xMinusW, 50);

    console.log(`Number of "b-a" calculations: ${bMinusA.length}`);
    console.log(`50th percentile (median) of "b-a": ${medianBA}`);
    console.log(`Number of "x-w" calculations: ${xMinusW.length}`);
    console.log(`50th percentile (median) of "x-w": ${medianXW}`);

    return {
        bMinusA: { median: medianBA, count: bMinusA.length },
        xMinusW: { median: medianXW, count: xMinusW.length }
    };
}

// Example usage with the file content
import * as fs from 'fs';
if (process.argv.length !== 3) {
    console.error('Usage: node calc_ts_data.js <logFilePath>');
    process.exit(1);
}
const fileContent = fs.readFileSync(process.argv[2], 'utf8');
const results = main(fileContent);