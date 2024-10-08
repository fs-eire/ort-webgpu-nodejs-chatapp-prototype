# Minimal example of using ONNX Runtime with WebGPU in Node.js

## Setup steps

1. build onnxruntime with WebGPU

   use branch `fs-eire/webgpu-ep`

   in the root folder of onnxruntime source code, run the following command:
   ```
   build --config Debug --use_webgpu --build_nodejs --skip_tests
   ```

3. make a symbol link to the root folder of ONNX Runtime source code. (assume the source code is in C:\code\onnxruntime)
   ```
   mklink /D /J onnxruntime C:\code\onnxruntime
   ```

4. run `npm install` in the root folder of this project.

5. prepare model.
   ```
   md models\microsoft
   cd models\microsoft
   git clone https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web
   ```
   if not working, try [this link](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx-web/tree/main?clone=true).

6. run `node .\main.js` to run the sample code.
