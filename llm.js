import * as ort from 'onnxruntime-node';
import * as path from 'node:path';
import * as url from 'node:url';
import * as fs from 'node:fs/promises';

ort.env.wasm.numThreads = 1;
ort.env.wasm.simd = true;
//const path = document.location.pathname.replace('ort-llm.html', '').replace('index.html', '').replace('local-chat.html', '');
//ort.env.wasm.wasmPaths = path + 'dist/';


function log(i) { console.log(i); /* document.getElementById('status').innerText += `\n${i}`; */ }

//
// load file from server or cache
//
async function fetchAndCache(url) {
    try {
        // const cache = await caches.open("onnx");
        // let cachedResponse = await cache.match(url);
        // if (cachedResponse === undefined) {
        log(`${url} (network)`);
        const buffer = await fs.readFile(url);
        //const buffer = await fetch(url).then(response => response.arrayBuffer());
        // try {
        //     await cache.put(url, new Response(buffer));
        // } catch (error) {
        //     console.error(error);
        // }
        return buffer;
        // }
        // log(`${url} (cached)`);
        // const data = await cachedResponse.arrayBuffer();
        // return data;
    } catch (error) {
        log(`can't fetch ${url}`);
        throw error;
    }
}

//
// class to handle a large language model on top of onnxruntime-web
//
export class LLM {
    sess = undefined;
    profiler = false;
    feed = {};
    output_tokens = [];
    eos = 2;
    need_position_ids = true;
    stop = false;
    kv_dims = [];
    dtype = "float16";
    max_tokens = 9999;
    trace = false;

    constructor() {
    }

    async load(model, options) {
        const provider = options.provider || "webgpu";
        const verbose = options.verbose;
        const trace = options.trace;
        const local = options.local;
        const hasFP16 = (provider === "wasm") ? false : options.hasFP16;
        this.profiler = options.profiler;

        const model_path = (local) ? "models/" + model.path : "https://huggingface.co/" + model.path + "/resolve/main";
        let model_file = model.file || "model";
        if (!model_file.endsWith(".onnx")) {
            model_file = (hasFP16) ? model_file + "_q4f16.onnx" : model_file + "_q4.onnx";
        }
        log(`loading... ${model.name},  ${provider}`);
        const json_bytes = await fetchAndCache(model_path + "/config.json");
        let textDecoder = new TextDecoder();
        const model_config = JSON.parse(textDecoder.decode(json_bytes));

        this.eos = model_config.eos_token_id;
        if (!Array.isArray(this.eos)) {
            this.eos = [this.eos];
        }
        this.eos.push(32007);
        this.eos = BigInt64Array.from(this.eos, (i) => BigInt(i));

        let model_bytes = await fetchAndCache(model_path + "/onnx/" + model_file);
        let externaldata = (model.externaldata) ? await fetchAndCache(model_path + "/onnx/" + model_file + '_data') : false;
        let modelSize = model_bytes.byteLength;
        if (externaldata) {
            modelSize += externaldata.byteLength;
        }
        log(`model size ${Math.round(modelSize / 1024 / 1024)} MB`);

        const opt = {
            executionProviders: [provider],
            preferredOutputLocation: {},
            intraOpNumThreads: 1,
            enableMemPattern: false,
            enableCpuMemArena: false,
            // optimizedModelFilePath: "opt.onnx",
            extra: {
                session: {
                    optimized_model_external_initializers_file_name: 'opt.onnx_data',
                    //optimized_model_external_initializers_min_size_in_bytes: '1048576', // 1024 * 1024
                }
            },
        }
        if (opt.executionProviders[0] === "webgpu") {
            opt.executionProviders[0] = {
                name: "webgpu",
                validationMode: 'wgpuOnly',
                storageBufferCacheMode: 'bucket',
                //forceCpuNodeNames: "/model/embed_tokens/Gather"
            };
        }

        switch (provider) {
            case "webgpu":
                for (let i = 0; i < model_config.num_hidden_layers; ++i) {
                    opt.preferredOutputLocation[`present.${i}.key`] = 'gpu-buffer';
                    opt.preferredOutputLocation[`present.${i}.value`] = 'gpu-buffer';
                }
                break;
        }

        if (externaldata !== undefined) {
            opt.externalData = [
                {
                    data: externaldata,
                    path: model_file + "_data",
                },
            ]
        }
        ort.env.webgpu.profiling = {}
        if (verbose) {
            opt.logSeverityLevel = 0;
            opt.logVerbosityLevel = 0;
            ort.env.logLevel = "verbose";
            if (verbose > 1) {
                ort.env.debug = true;
                ort.env.webgpu.profiling.mode = 'default';
            }
        }

        if (this.profiler) {
            opt.enableProfiling = true;
            opt.profileFilePrefix = model.name;
            //ort.env.webgpu.profilingMode = 'default';
            //ort.env.webgpu.profiling.mode = 'default';
        }
        if (trace) {
            ort.env.trace = true;
            //ort.env.webgpu.profiling.ondata =
            //    (version, inputsMetadata, outputsMetadata, kernelId, kernelType, kernelName, programName, startTime, endTime) => { };
        }

        // opt.optimizedModelFilePath = 'opt.onnx';
        // opt.graphOptimizationLevel = "disabled";

        this.sess = await ort.InferenceSession.create(model_bytes, opt);
        const dim_kv = model_config.head_dim || model_config.hidden_size / model_config.num_attention_heads;
        this.kv_dims = [1, model_config.num_key_value_heads, 0, dim_kv];
        this.dtype = (hasFP16) ? "float16" : "float32";
        this.num_layers = model_config.num_hidden_layers;
        model_bytes = undefined;
        externaldata = false;
        this.initilize_feed();
    }

    initilize_feed() {
        const feed = this.feed;

        // dispose of previous gpu buffers
        for (const name in feed) {
            const t = feed[name];
            if (t.location === 'gpu-buffer') {
                t.dispose();
            }
        }
        this.feed = {};
        // key value cache is zero copy, just pass gpu buffer as referece
        const empty = (this.dtype === "float16") ? new Uint16Array() : [];
        for (let i = 0; i < this.num_layers; ++i) {
            this.feed[`past_key_values.${i}.key`] = new ort.Tensor(this.dtype, empty, this.kv_dims)
            this.feed[`past_key_values.${i}.value`] = new ort.Tensor(this.dtype, empty, this.kv_dims)
        }
        this.output_tokens = [];
    }

    //
    // poor mens argmax
    argmax(t) {
        const arr = t.data;
        const start = t.dims[2] * (t.dims[1] - 1);
        let max = arr[start];
        let maxidx = 0;

        for (let i = 0; i < t.dims[2]; i++) {
            const val = arr[i + start];
            if (isFinite(val)) {
                if (val > max) {
                    max = arr[i + start];
                    maxidx = i;
                }
            } else {
                throw new Error("found infinitive in logits");
            }
        }
        return maxidx;
    }

    //
    // update key value cache
    //
    update_kv_cache(feed, outputs) {
        for (const name in outputs) {
            if (name.startsWith('present')) {
                let newName = name.replace('present', 'past_key_values');
                // dispose previous gpu buffers
                const t = feed[newName];
                if (t.location === 'gpu-buffer') {
                    t.dispose();
                }
                feed[newName] = outputs[name];
            }
        }
    }

    //
    // tell generate to stop()
    //
    abort() {
        this.stop = true;
    }

    webgpu_tensor_from_tensor(oldt, t) {
        return t;
        /*
        oldt.dispose();
        const device = ort.env.webgpu.device;
        const size = Math.ceil(t.data.byteLength / 64) * 64;
        const gpubuf = device.createBuffer({ mappedAtCreation: true, size: size, usage: GPUBufferUsage.COPY_SRC|GPUBufferUsage.MAP_WRITE });
        const arr = gpubuf.getMappedRange();
        //new Uint8Array(arr).set(t.data.buffer);
        new Uint8Array(arr).set(new Uint8Array(t.data.buffer, t.data.byteOffset, t.data.byteLength));

        gpubuf.unmap();
        return ort.Tensor.fromGpuBuffer(gpubuf, { dataType: t.type, dims: t.dims });
        */
    }

    //
    // prefill prompt and generate tokens
    //
    async generate(tokens, callback, options) {
        // xdf();
        const feed = this.feed;
        const input_ids = new ort.Tensor('int64', BigInt64Array.from(tokens.map(BigInt)), [1, tokens.length]);
        feed['input_ids'] = input_ids;
        this.stop = false;

        this.output_tokens.push(...input_ids.data);

        let last_token = -1n;
        let seqlen = this.output_tokens.length;
        const input_len = input_ids.size;
        const max_tokens = (options.max_tokens || 256) + input_len;

        if (this.need_position_ids) {
            feed['position_ids'] = new ort.Tensor('int64', BigInt64Array.from({ length: input_len }, (_, i) => BigInt(seqlen - input_len + i)), [1, input_len]);
        }

        let xxi = 0;
        while (!this.eos.includes(last_token) && seqlen < max_tokens && !this.stop) {
            seqlen = this.output_tokens.length;
            feed['attention_mask'] = new ort.Tensor('int64', BigInt64Array.from({ length: seqlen }, () => 1n), [1, seqlen]);
            //const pending = this.sess.run(feed);
            const outputs = await this.sess.run(feed);
            // if (xxi++ > 32) {
            //     this.sess.endProfiling();

            //     xdf();
            // }
            last_token = BigInt(this.argmax(outputs.logits));
            this.output_tokens.push(last_token);
            if (callback /* && !this.profiler */) {
                callback(this.output_tokens);
            }
            this.update_kv_cache(feed, outputs);
            feed['input_ids'] = new ort.Tensor('int64', BigInt64Array.from([last_token]), [1, 1]);
            if (this.need_position_ids) {
                feed['position_ids'] = new ort.Tensor('int64', BigInt64Array.from([BigInt(seqlen + 1)]), [1, 1]);
            }
        }
        if (this.profiler) {
            this.sess.endProfiling();
        }
        return this.output_tokens;
    }
}
