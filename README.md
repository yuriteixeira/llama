# Llama.cpp - My experiments with models & presets

## Testing Workflow

Assuming you cloned to `/home/you/repos/llama` and that you use https://pi.dev: 

1. Write preset entries on `./model-presets.upstream.ini` based on the hugging face model name (eg: `unsloth/Qwen3.8-27B-UD-IQ4_XS.gguf`) pointing to the absolute path of the model's download path (eg: `/home/you/repos/llama/models/unsloth/Qwen3.8-27B-UD-IQ4_XS.gguf`). There are examples, so just copy/paste/adapt.
2. Run `./manage-server.sh download-models`, which will download the models from https://huggingface.co.
3. Run `./manage-server.sh restart && ./manage-server.sh logs`
4. Use Pi's `/llama` to load a model (if it fails to load, logs will tell why, probably OOM or unsupported args)
5. Test it out!

## About the LLM inference engines inside `./versions`

We're focusing on `upstream` llama-cpp for now, but we tested other forks like `turboquant` & `buun` in the past. Check git logs for more details.
They a

### Updating the inference engines

To fetch the latest upstream on the `master` branch and compile it, run:

```sh
./manage-server.sh update-build
```

Select a fork or a different branch with optional arguments:

```sh
./manage-server.sh update-build buun
./manage-server.sh update-build turboquant
./manage-server.sh update-build upstream master
```


