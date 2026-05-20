from default_config.DefaultFunctionConfig import *

# ----- Recall -----
DEFAULT_FUMEMORY_RECALL = {
    'method': 'FUMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_UTILIZATION,
    'empty_memory': 'None'
}

DEFAULT_STMEMORY_RECALL = {
    'method': 'STMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_UTILIZATION,
    'time_retrieval': DEFAULT_TIME_RETRIEVAL,
    'empty_memory': 'None'
}

DEFAULT_LTMEMORY_RECALL = {
    'method': 'LTMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_UTILIZATION,
    'text_retrieval': DEFAULT_TEXT_RETRIEVAL,
    'empty_memory': 'None'
}

DEFAULT_GAMEMORY_RECALL = {
    'method': 'GAMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_UTILIZATION,
    'text_retrieval': DEFAULT_TEXT_RETRIEVAL,
    'time_retrieval': {
        'method': 'TimeRetrieval',
        'mode': 'exp',
        'coef': {
            'decay': 0.995
        }
    },
    'importance_retrieval': DEFAULT_VALUE_RETRIEVAL,
    'importance_judge': DEFAULT_IMPORTANCE_JUDGE,
    'topk': 5,
    'empty_memory': 'None'
}

DEFAULT_MBMEMORY_RECALL = {
    'method': 'MBMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_UTILIZATION,
    'text_retrieval': DEFAULT_TEXT_RETRIEVAL,
    'empty_memory': 'None',
    'forget': DEFAULT_FORGET
}

DEFAULT_SCMEMORY_RECALL = {
    'method': 'SCMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_UTILIZATION,
    'text_retrieval': DEFAULT_TEXT_RETRIEVAL,
    'time_retrieval': {
        'method': 'TimeRetrieval',
        'mode': 'exp',
        'coef': {
            'decay': 0.995
        }
    },
    'empty_memory': 'None',
    'flash_capacity': 3,
    'activation_topk': 3,
    'activation_judge': {
        'method': 'LLMJudge',
        'LLM_config': DEFAULT_LLM,
        'prompt': {
            'template': """Query: {query}
Memory: {flash_memory}
Given the above memory, determine whether it requires other information to generate correct response for the query.
You should just output True or False, without any other messages.""",
            'input_variables': ['query', 'flash_memory']
        }
    },
    'summary_judge': {
        'method': 'LLMJudge',
        'LLM_config': DEFAULT_LLM,
        'prompt': {
            'template': """Query: {query}
Memory: {activation_summary}
{flash_memory}
Based on the memory, can the query be responsed correctly?
You should just output True or False, without any other messages.""",
            'input_variables': ['query', 'activation_memory', 'flash_memory']
        }
    },
    'summarizer': DEFAULT_SUMMARIZER
}

DEFAULT_MGMEMORY_RECALL = {
    'method': 'MGMemoryRecall',
    'truncation': DEFAULT_TRUNCATION,
    'utilization': DEFAULT_UTILIZATION,
    'warning_threshold': 0.7,
    'warning_content': 'We suggest to execute memory_archive or memory_transfer.',
    'trigger': DEFAULT_MGMEMORY_LLMTRIGGER,
    'recall_retrieval': DEFAULT_TEXT_RETRIEVAL,
    'archival_retrieval': DEFAULT_TEXT_RETRIEVAL,
    'empty_memory': 'None'
}

DEFAULT_RFMEMORY_RECALL = DEFAULT_FUMEMORY_RECALL

# ----- Store -----
DEFAULT_FUMEMORY_STORE = {
    'method': 'FUMemoryStore'
}

DEFAULT_STMEMORY_STORE = {
    'method': 'LTMemoryStore'
}

DEFAULT_LTMEMORY_STORE = {
    'method': 'LTMemoryStore'
}

DEFAULT_GAMEMORY_STORE = {
    'method': 'GAMemoryStore'
}

DEFAULT_MBMEMORY_STORE = {
    'method': 'MBMemoryStore',
    'summarizer': DEFAULT_SUMMARIZER
}

DEFAULT_SCMEMORY_STORE = {
    'method': 'SCMemoryStore'
}

DEFAULT_MGMEMORY_STORE = {
    'method': 'MGMemoryStore',
    'flush_checker': DEFAULT_TRUNCATION,
    'summarizer': {
        'method': 'LLMSummarizer',
        'LLM_config': DEFAULT_LLM,
        'prompt': {
            'template': """Recursive Summary: {recursive_summary}
Recent Memory: {flush_context}
Please update the Recursive Summary based on Recursive Summary and summarizing Recent Memory.
Just output the new Recursive Summary, without any other messages.""",
            'input_variables': ['recursive_summary', 'flush_context']
        }
    }
}

DEFAULT_MTMEMORY_STORE = {
    'method': 'MTMemoryStore',
    'traverse_base_threshold': 0.4,
    'traverse_rate': 0.5,
    'summarizer': {
        'method': 'LLMSummarizer',
        'LLM_config': DEFAULT_LLM,
        'prompt': {
            'template': """You will receive two pieces of information: New Information is detailed, and Existing Information is a summary from {n_children} previous entries. Your task is to merge these into a single, cohesive summary that highlights the most important insights.
- Focus on the key points from both inputs.
- Ensure the final summary combines the insights from both pieces of information.
- If the number of previous entries in Existing Information is accumulating (more than 2), focus on summarizing more concisely, only capturing the overarching theme, and getting more abstract in your summary.  Output the summary directly. 
[New Information] {new_content}
[Existing Information (from {n_children} previous entries)
{current_content}
[ Output Summary ]""",
            'input_variables': ['n_children', 'new_content', 'current_content']
        }
    }
}

# ----- Reflect -----
DEFAULT_GAMEMORY_REFLECT = {
    'method': 'GAReflect',
    'reflector': DEFAULT_GAREFLECTOR
}

# ----- Optimize -----
DEFAULT_RFMEMORY_OPTIMIZE = {
    'method': 'RFMemoryOptimize',
    'reflector': DEFAULT_TRIALREFLECTOR
}