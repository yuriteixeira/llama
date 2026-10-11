# Llama.cpp - My experiments with models & presets

## Disclaimer

This repo have no intention to be a generic solution. Instead, it hopes to serve as a base for you to fork, adapt and start experimenting.

Use AI to help you out!

## Workflow

Assuming you cloned to `/home/you/repos/llama` and that you use https://pi.dev: 

1. Research what models fit your hardware and write preset entries accordingly in `./model-presets.upstream.ini` (just follow the examples there)
2. Run `./manage-server.sh download-models`, which will download the models from https://huggingface.co.
3. Run `./manage-server.sh build`, and then, `./manage-server.sh enable && ./manage-server.sh start && ./manage-server.sh logs`. Test with `curl -s http://localhost:8080/v1/models`, it should return the preset entries you have.
4. Use Pi's `/llama` to load a model (if it fails to load, logs will tell why, probably OOM or unsupported args)
5. Prompt the hell out of it and start your tweaking journey!

## About the LLM inference engines inside `./versions`

We're focusing on `upstream` llama-cpp for now, but we tested other forks like `turboquant` & `buun` in the past. Check git logs for more details.
They a

### Updating the inference engines

To fetch the latest upstream on the `master` branch and compile it, run:

```sh
./manage-server.sh changelog    # To see what's changed before you update (depends on Pi configured with a working model)
./manage-server.sh update-build # Point engine's submodule to the latest remote commit and re-build it
```
