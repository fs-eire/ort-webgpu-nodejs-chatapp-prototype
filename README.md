# Minimal example of using ONNX Runtime with WebGPU in Node.js

## Setup steps

1. build onnxruntime with WebGPU

   in the root folder of onnxruntime source code, run the following command:
   ```
   build --config Debug --use_webgpu --build_nodejs --skip_tests
   ```

2. make a symbol link to the root folder of ONNX Runtime source code. (assume the source code is in C:\code\onnxruntime)
   ```
   mklink /D /J onnxruntime C:\code\onnxruntime
   ```

3. run `npm install` in the root folder of this project.

4. prepare model.
   ```
   md models\microsoft
   cd models\microsoft
   git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web
   ```
   if not working, try [this link](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web/tree/main?clone=true).

5. run `node .\main.js` to run the sample code.