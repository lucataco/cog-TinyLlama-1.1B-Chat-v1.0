# TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF Cog model

This is an implementation of [TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) as a [Cog](https://github.com/replicate/cog) model.

## Development

Follow the [model pushing guide](https://replicate.com/docs/guides/push-a-model) to push your own model to [Replicate](https://replicate.com).

## Basic Usage

Download the weights

    cog run bash download-weights.sh

Run a prediction

    cog predict -i prompt="How many helicopters can a human eat in one sitting?"

