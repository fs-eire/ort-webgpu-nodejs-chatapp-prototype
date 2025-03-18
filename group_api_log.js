import * as fs from 'fs';
import * as readline from 'readline';

// Usage check
if (process.argv.length < 3) {
    console.error('Usage: node group_api_log.js <input_file>');
    process.exit(1);
}

const inputFile = process.argv[2];
const lineCounts = new Map();

// Create readline interface
const processFile = async () => {
    try {
        const fileStream = fs.createReadStream(inputFile);
        const rl = readline.createInterface({
            input: fileStream,
            crlfDelay: Infinity
        });

        let total_lines = 0;

        let flag = false;

        // Process each line
        for await (const line of rl) {

            if (line.startsWith('##')) {
                total_lines++;
                const flaged_line = (flag ? 's ' : '  ') + line;
                // Update count for this line
                lineCounts.set(flaged_line, (lineCounts.get(flaged_line) || 0) + 1);
            } else if (line.trim() === '$$ submit off') {
                flag = false;
            } else if (line.trim() === '$$ submit on') {
                flag = true;
            }
        }

        // Convert to array, sort by count descending, and output
        const sortedResults = Array.from(lineCounts.entries())
            .sort((a, b) => b[1] - a[1])
            .map(([line, count]) => ({ line, count }));

        console.log('Line occurrence counts (sorted by frequency):');
        console.log('-------------------------------------------');
        sortedResults.forEach(item => {
            console.log(`${((item.count / total_lines) * 100).toFixed(2)}%\t${item.count}\t${item.line}`);
        });

        console.log('\nTotal unique lines:', sortedResults.length);
    } catch (err) {
        console.error('Error processing file:', err);
    }
};

processFile();