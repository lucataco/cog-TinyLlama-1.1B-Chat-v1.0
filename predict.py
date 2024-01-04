from cog import BasePredictor, Input, ConcatenateIterator
from llama_cpp import Llama

PROMPT_TEMPLATE = "<|system|>\n{system_prompt}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"
SYSTEM_PROMPT = "You are a friendly chatbot who always responds in the style of a pirate"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = Llama(
            model_path="./tinyllama-1.1b-chat-v1.0.Q5_K_M.gguf",
            n_ctx=2048,
            n_threads=8,
            n_gpu_layers=35,
        )

    def predict(
        self,
        prompt: str = Input(
            description="Instruction for model",
            default="How many helicopters can a human eat in one sitting?"
        ),
        system_prompt: str = Input(
            description="System prompt for the model, helps guides model behaviour.",
            default=SYSTEM_PROMPT,
        ),
        prompt_template: str = Input(
            description="Template to pass to model. Override if you are providing multi-turn instructions.",
            default=PROMPT_TEMPLATE,
        ),
        max_new_tokens: int = Input(
            description="The maximum number of tokens the model should generate as output", default=256
        ),
        top_p: float = Input(
            description="This parameter controls how many of the highest-probability words are selected to be included in the generated text",
            default=0.95,
        ),
        top_k: int = Input(
            description="The number of highest probability tokens to consider for generating the output. If > 0, only keep the top k tokens with highest probability (top-k filtering)",
            default=50,
        ),
        temperature: float = Input(
            description="The value used to modulate the next token probabilities.",
            default=0.7,
        ),
    ) -> ConcatenateIterator[str]:
        """Run a single prediction on the model"""

        full_prompt = prompt_template.replace("{prompt}", prompt).replace(
            "{system_prompt}", system_prompt
        )

        for output in self.model(
            full_prompt,
            stream=True,
            top_p=top_p,
            top_k=top_k,
            max_tokens=max_new_tokens,
            temperature=temperature,
        ):
            yield output["choices"][0]["text"]