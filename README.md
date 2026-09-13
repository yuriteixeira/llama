# Llama.cpp - My experiments with models & presets

## Testing Workflow

1. Choose a model
2. Set-up a preset for that model and let `./scripts/download-preset-models.sh` to download and save them in the `./models/` folder.
3. Run `./manage-server.sh restart && ./manage-server.sh logs`
4. Use Pi's `/llama` to load a model (if it fails to load, logs will tell why, probably OOM or unsupported args)
5. Test it out!

## About versions

We have for now `upstream` (vanilla's llama-cpp) and some forks like `turboquant` & `buun`. Another one can always be added for test purposes, but for now, we're focusing on regular llama-cpp.

