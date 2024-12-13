import { LLM } from './llm.js';

import * as ort from 'onnxruntime-node';
// must import onnxruntime-node before transformerjs, otherwise it will
// use the onnxruntime-node@1.14.0 required by transformerjs instead of
// the latest.
//
// transformerjs is only used for tokenization.
import { AutoTokenizer, env } from '@xenova/transformers';

function log(i) { console.log(i); document.getElementById('status').innerText += `\n${i}`; }

const MODELS = {
    "smollm-360": { name: "smollm-360", path: "Xenova/SmolLM-360M", file: "model_q4.onnx", fp16: false },
    "tinyllama": { name: "tinyllama", path: "schmuell/TinyLlama-1.1B-Chat-v1.0-int4", file: "decoder_model_merged" },
    "tinyllama_fp16": { name: "tinyllama-fp16", path: "schmuell/TinyLlama-1.1B-Chat-v1.0-fp16", externaldata: true, file: "decoder_model_merged" },
    "phi2": { name: "phi2", path: "schmuell/phi2-int4", file: "decoder_model_merged" },
    "phi3": { name: "phi3", path: "microsoft/Phi-3-mini-4k-instruct-onnx-web", externaldata: true },
    "phi3-1": { name: "phi3-1", path: "schmuell/phi3-1", externaldata: true },
    "stablelm": { name: "stablelm", path: "schmuell/stablelm-2-zephyr-1_6b-int4", file: "decoder_model_merged" },
}

const SUM = `Summarize:
Constantinople, now known as Istanbul in modern Turkey, was a historically significant city that served as the capital of both the Roman / Byzantine
Empire and the Ottoman Empire.Its rich history spans over 2, 500 years, with its strategic location at the crossroads between Europe and Asia contributing
to its prominence throughout various periods.The city was originally founded by Greek colonists from Megara as Byzantium around 657 BC.It became a
significant center of trade due to its position on the Bosphorus, controlling passage between the Black Sea and the Mediterranean.However, it gained
even greater importance after Emperor Constantine I(Constantinus) relocated his capital there in 324 AD, thus renaming it Constantinople.
The Byzantine Empire developed into a major hub of Christian culture and religion, with Hagia Sophia being one of its most iconic structures
built during the reign of Emperor Justinian I.The city flourished as an artistic and intellectual center until the 12th century when it faced
significant challenges from various invaders, including Arabs, Bulgarians, Crusaders, and Venetians.
    In 1453, Constantinople fell to the Ottoman Empire after a protracted siege led by Sultan Mehmed II.The city was renamed Istanbul as part of the empire's
policy of Islamization, but it retained much of its Greek Byzantine culture and architecture under the new rule.Today, Istanbul is Turkey's largest city and
an important cultural, economic, and transportation hub.The historical significance of Constantinople / Istanbul lies in its architectural landmarks
such as Hagia Sophia, The Hippodrome(now Sultanahmet Square), the Chora Church, and many more that showcase a blend of Byzantine, Roman, and Ottoman influences.
`;

const TASK = {
    "sum": SUM,
    "easy": "Tell me about Constantinople.",
}

function getConfig() {
    //const query = window.location.search.substring(1);
    var config = {
        model: "phi3",
        provider: "webgpu",
        profiler: 0,
        verbose: 0,
        threads: 1,
        trace: 0,
        csv: 0,
        max_tokens: 300,
        local: 1,
        values: 0,
        task: "sum",
    }
    // let vars = query.split("&");
    // for (var i = 0; i < vars.length; i++) {
    //     let pair = vars[i].split("=");
    //     if (pair[0] in config) {
    //         const key = pair[0];
    //         const value = decodeURIComponent(pair[1]);
    //         if (typeof config[key] == "number") {
    //             config[key] = parseInt(value);
    //         }
    //         else {
    //             config[key] = value;
    //         }
    //     } else if (pair[0].length > 0) {
    //         throw new Error("unknown argument: " + pair[0]);
    //     }
    // }
    if (MODELS[config.model] !== undefined) {
        config.model = MODELS[config.model];
    }
    return config;
}

const config = getConfig();
env.localModelPath = 'models';
env.allowRemoteModels = config.local == 0;
env.allowLocalModels = config.local == 1;
// ort.env.wasm.numThreads = config.threads;
// ort.env.wasm.simd = true;

const cons_log = [];

function redirect_output() {
    console.log = function (message) {
        if (!message.includes('_fence_')) {
            cons_log.push(message);
        }
    };
}

if (config.profiler === 2) {
    redirect_output();
}

const tokenizer = await AutoTokenizer.from_pretrained(config.model.path);

function create_download_link(cons_log) {
    if (cons_log.length > 0) {
        let link = document.getElementById('download').childNodes[0];
        if (link === undefined) {
            link = document.createElement("a", "download-link");
            link.download = "profiler.log";
            link.innerText = "Download";
            document.getElementById('download').appendChild(link);
        }
        const base64 = btoa(cons_log.join('\n'));
        link.href = `data:application/json;base64,${base64}`;
    }
}


function token_to_text(tokenizer, tokens, startidx) {
    const txt = tokenizer.decode(tokens.slice(startidx), { skip_special_tokens: true, });
    return txt;
}

const llm = new LLM();

async function main() {

    // NODE.js I/O hack
    var OUTPUT_RESULT = (txt) => {
        console.log(txt);
    };
    //

    const model = config.model;

    await llm.load(model, {
        provider: config.provider,
        verbose: config.verbose,
        profiler: config.profiler,
        trace: config.trace,
        local: config.local,
        hasFP16: (config.model.fp16 != undefined) ? config.model.fp16 : true,
    });


    //document.getElementById('status').innerText = "";
    const query = TASK[config.task];
    const prompt = `<|system|>\nYou are a friendly assistant.<|end|>\n<|user|>\n${query}<|end|>\n<|assistant|>\n`;
    const { input_ids } = await tokenizer(prompt, { return_tensor: false, padding: true, truncation: true });

    const start_timer = performance.now();

    let look_back_tokens = [];
    const display_callback = (last_token) => {
        const look_back_string = look_back_tokens.length === 0 ? '' : tokenizer.decode(look_back_tokens, { skip_special_tokens: true, clean_up_tokenization_spaces: true });
        const new_string = tokenizer.decode([...look_back_tokens, last_token], { skip_special_tokens: true, clean_up_tokenization_spaces: true });
        const new_word = new_string.slice(look_back_string.length);
        process.stdout.write(new_word);
        look_back_tokens[0] = last_token;
    };

    const output_tokens = await llm.generate(input_ids, display_callback, { max_tokens: config.max_tokens, values: config.values });
    const end_time = performance.now();
    const took = (end_time - start_timer) / 1000;
    const firstTokenDecodingTime = (llm.firstTokenDoneTime - start_timer) / 1000;
    const remainingTokensDecodingTime = (end_time - llm.firstTokenDoneTime) / 1000;
    const txt = token_to_text(tokenizer, output_tokens, input_ids.length);
    const seqlen = output_tokens.length;
    OUTPUT_RESULT(txt);
    const perf = `${seqlen} tokens in ${took.toFixed(1)}sec, ${(seqlen / took).toFixed(2)} tokens/sec
    Decoding first token with input ${llm.promptTokens} tokens: ${firstTokenDecodingTime.toFixed(1)} sec
    Decoding remaining ${seqlen - llm.promptTokens} tokens:
\t${remainingTokensDecodingTime.toFixed(1)} sec
\t${((seqlen - llm.promptTokens) / remainingTokensDecodingTime).toFixed(2)} tokens/sec
    `;
   // console.log(perf + " @@1");
    OUTPUT_RESULT(perf);
    if (config.csv) {
        log(`${model.name},${took.toFixed(2)},${(seqlen / took).toFixed(3)},${seqlen},@@2`);
    }
}
try {
    await main();
} catch (error) {
    console.error(error);
    console.error(error.message);
} finally {
    //create_download_link(cons_log);
}
