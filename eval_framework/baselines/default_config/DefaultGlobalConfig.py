import os as _os

DEFAULT_GLOBAL_CONFIG = {
    'usable_gpu': _os.getenv('CUDA_VISIBLE_DEVICES', '0,1')
}

# --- Top-K defaults shared by upstream MM baselines ---
DEFAULT_RETRIEVAL_TOP_K = 10      # text + multimodal retrieval
DEFAULT_GRAPH_MAX_NODES = 10      # NGMemory + AUGUSTUSMemory traversal
DEFAULT_REFLECTION_TOP_K = 10     # GAReflector

# --- API + model paths ---
DEFAULT_OPENAI_APIKEY = _os.getenv('OPENAI_API_KEY', '')
DEFAULT_OPENAI_APIBASE = _os.getenv('OPENAI_BASE_URL', '')

DEFAULT_LLAMA3_8B_INSTRUCT_PATH = _os.getenv('LLAMA3_8B_INSTRUCT_PATH', '')
DEFAULT_E5_BASE_V2_PATH = _os.getenv('E5_BASE_V2_PATH', '')

# Multimodal encoder paths used by upstream MM configs.
DEFAULT_GME_QWEN2_VL_2B_PATH = _os.getenv(
    'GME_MODEL_PATH', 'Alibaba-NLP/gme-Qwen2-VL-2B-Instruct'
)
DEFAULT_BACKBONE_PATH = _os.getenv('LLM_BACKBONE_PATH', '')