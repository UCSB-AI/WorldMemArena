# M²A: Multimodal Memory Agent with Dual-Layer Hybrid Memory for Long-Term Personalized Interactions


## 1. Installation

### Quick Start

```bash
# Create virtual environment (using uv)
uv venv

# Activate environment
source .venv/bin/activate

# Install dependencies
uv sync
```

## 2. Configuration

The system supports flexible configuration through environment variables or a TOML config file.

### Quick Configuration

For quick testing, you can set key environment variables:

```bash
export M2A_LLM_MODEL="gpt-4o-mini"
export M2A_LLM_BASE_URL="https://api.openai.com/v1"
export M2A_LLM_API_KEY="sk-..."

export M2A_TEXT_EMBEDDING_MODEL="all-MiniLM-L6-v2"
export M2A_TEXT_EMBEDDING_BASE_URL="http://localhost:8010/v1" # local serving
export M2A_TEXT_EMBEDDING_API_KEY="EMPTY"

export M2A_MULTIMODAL_EMBEDDING_MODEL="siglip2-base-patch16-384"
export M2A_MULTIMODAL_EMBEDDING_BASE_URL="http://localhost:8050/v1" # local serving
export M2A_MULTIMODAL_EMBEDDING_API_KEY="EMPTY"
```

### Using a Config File

For more complex configurations, use a TOML config file. Please refer to `agent/config.py` for all supported configurations.

## 3. Data Preparation

### Download Datasets

Please download the LoCoMo dataset from [here](https://github.com/snap-research/LoCoMo) and the YoLLaVA dataset from [here](https://github.com/WisconsinAIVision/YoLLaVA).


### Preprocessing

To construct the multimodally enhanced LoCoMo dataset, please refer to `data/yollava_generate_and_merge.py`.

This script injects visual-centric QA pairs from YoLLaVA into LoCoMo following the pipeline described in the paper.

## 4. Evaluation

The system includes an evaluation wrapper for running experiments on the enhanced LoCoMo dataset.

### Running Evaluation
The `eval_wrapper.py` module provides a wrapper class that implements the interface expected by `data/evaluator.py`:

- `start_conversation(conv_info)` - Initialize for a new conversation
- `chat(dialogue)` - Process dialogue history (builds memory)
- `question(text, image)` - Answer a question about the conversation

Here is an example using the evaluator:
```python
from argparse import ArgumentParser
import json

from eval_wrapper import M2AEvaluationWrapper
from agent.config import M2AConfig
from eval.llm_judge import LLMJudge
from eval.evaluator import Evaluator


def parse_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--n_parallel", type=int)

    return arg_parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = M2AConfig.from_file("config.toml")
    models = [M2AEvaluationWrapper(config) for _ in range(args.n_parallel)]

    judge = LLMJudge(
        base_url="https://api.openai.com/v1",
        api_key="sk-...",
        model="gpt-4o-mini"
    )
    evaluator = Evaluator(models, judge, database_root_path="./dataset")
    results = evaluator.evaluate_file("./dataset/eval_dataset.json")

    with open("result.json", 'w') as f:
        json.dump(results, f)
```

This evaluation protocol follows the setup described in the paper.

## License

This project is licensed under the MIT License.