import re
import json
import math
import threading
from tqdm import tqdm
from queue import Queue
from typing import Any, TypedDict
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from .llm_judge import LLMJudge

CATEGORIES = {
    1: "Multi-hop", 2: "Temporal",
    3: "Open-domain", 4: "Single-hop", 5: "VQA",
    6: "V-Multi-hop", 7: "V-Temporal",
    8: "V-Open-domain", 9: "V-Single-hop",
}

class Summary(TypedDict):
    total_questions: int
    overall: dict[str, float]
    by_category: dict[str, Any]


class Evaluator:
    """
    Evaluates model performance on conversation-based QA tasks using
    F1, BLEU, and LLM-based judging.
    """

    def __init__(
        self,
        methods: list,
        judge: LLMJudge,
        database_root_path: str
    ):
        self.judge = judge
        self.database_root_path = database_root_path
        self.method_pool = Queue()
        for m in methods:
            self.method_pool.put(m)
        self.stats_lock = threading.Lock()

    # --- Metric Calculations ---

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenizes string into lowercase words and CJK characters."""
        text = str(text).lower()
        text = re.sub(r'[\u0000-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E]', ' ', text)
        return re.findall(r'[\u4e00-\u9fff]|[a-z0-9]+', text)

    def calculate_f1(self, pred: str, ref: str) -> float:
        pt, rt = self.tokenize(pred), self.tokenize(ref)
        if not pt and not rt: return 1.0
        if not pt or not rt: return 0.0
        pc, rc = Counter(pt), Counter(rt)
        overlap = sum((pc & rc).values())
        if overlap == 0: return 0.0
        precision = overlap / len(pt)
        recall = overlap / len(rt)
        return 2 * precision * recall / (precision + recall)

    def calculate_bleu1(self, pred: str, ref: str) -> float:
        pt, rt = self.tokenize(pred), self.tokenize(ref)
        if not pt: return 0.0
        pc, rc = Counter(pt), Counter(rt)
        clipped_overlap = sum(min(pc[w], rc[w]) for w in pc)
        precision = clipped_overlap / len(pt)
        bp = math.exp(1 - len(rt) / len(pt)) if len(pt) < len(rt) else 1.0
        return bp * precision

    def _get_best_metrics(self, pred: str, refs: list[str]) -> tuple[float, float, str]:
        best_f1, best_bleu, best_ref = -1, -1, ""
        for r in refs:
            f1 = self.calculate_f1(pred, r)
            bleu = self.calculate_bleu1(pred, r)
            if (f1 + bleu) > (best_f1 + best_bleu):
                best_f1, best_bleu, best_ref = f1, bleu, r
        return best_f1, best_bleu, best_ref

    # --- Data Normalization ---

    def _norm_imgs(self, imgs: Any) -> list:
        if imgs is None: return []
        return [imgs] if isinstance(imgs, (str, dict)) else list(imgs)

    def _norm_refs(self, refs: Any) -> list[str]:
        if refs is None: return [""]
        if isinstance(refs, (str, int, float, bool)): return [str(refs)]
        items = [refs] if isinstance(refs, dict) else refs
        out = []
        for x in items:
            if isinstance(x, dict):
                val = x.get("text") or x.get("answer") or x.get("value") or ""
                out.append(str(val))
            else:
                out.append(str(x))
        return out or [""]

    # --- Data Collection ---

    def _build_dialogue(self, conversation: dict) -> list[dict]:
        """Flattens session-based conversation into a chronological list."""
        dialogue = []
        idx = 0
        while True:
            key = f"session_{idx}"
            if key not in conversation:
                break
            utterances = conversation[key]
            if isinstance(utterances, list):
                timestamp = conversation.get(f"session_{idx}_date_time", "")
                for u in utterances:
                    dialogue.append({
                        "speaker": u["speaker"],
                        "dia_id": u.get("dia_id", ""),
                        "images": self._norm_imgs(u.get("images", [])),
                        "text": u.get("text", ""),
                        "timestamp": timestamp
                    })
            idx += 1
        return dialogue

    def _collect_conversations(self, data: list[dict]) -> list[dict]:
        processed = []
        for i, item in enumerate(data):
            qas = []
            for j, qa in enumerate(item.get("qa", [])):
                q = qa["question"]
                qas.append({
                    "qid": f"{i}:{j}",
                    "question": q["text"],
                    "images": self._norm_imgs(q.get("image", [])),
                    "refs": self._norm_refs(qa.get("answers") or qa.get("answer")),
                    "category": qa.get("category", "default")
                })
            processed.append({
                "cid": i,
                "dialogue": self._build_dialogue(item["conversation"]),
                "qas": qas,
                "conv_info": {
                    "speaker_0": item['conversation'].get("speaker_0"),
                    "speaker_1": item['conversation'].get("speaker_1")
                }
            })
        return processed

    # --- Summary Calculation ---

    def calc_summary(self, results: dict, done: int) -> dict:
        """Aggregates metrics into a structured summary."""
        all_items = [x for items in results.values() for x in items]
        judged_items = [x for x in all_items if x['judge_label'] is not None]

        overall = {
            "F1": sum(x['f1'] for x in all_items) / len(all_items) if all_items else 0.0,
            "BLEU1": sum(x['bleu1'] for x in all_items) / len(all_items) if all_items else 0.0,
            "LLM_JUDGE": (
                sum(x['judge_label'] == "CORRECT" for x in judged_items) / len(judged_items)
                if judged_items else 0.0
            )
        }

        by_category = {}
        for cat, items in results.items():
            cat_judged = [x for x in items if x['judge_label'] is not None]
            by_category[cat] = {
                "count": len(items),
                "metrics": {
                    "F1": sum(x['f1'] for x in items) / len(items),
                    "BLEU1": sum(x['bleu1'] for x in items) / len(items),
                    "LLM_JUDGE": (
                        sum(x['judge_label'] == "CORRECT" for x in cat_judged) / len(cat_judged)
                        if cat_judged else 0.0
                    ),
                }
            }

        summary_obj: Summary = {
            "total_questions": done,
            "overall": overall,
            "by_category": by_category
        }
        return {"results": results, "summary": summary_obj}

    # --- Execution Logic ---

    def _worker(self, conv_idx, conv, max_samples, n_sample_msg):
        method = self.method_pool.get()
        try:
            conv["conv_info"]["conv_idx"] = conv_idx
            method.start_conversation(conv_info=conv["conv_info"])
            method.chat(conv["dialogue"][:n_sample_msg])

            qas = conv["qas"][:max_samples] if max_samples else conv["qas"]
            local_results = {}

            for qa in tqdm(qas, desc=f"[conv: {conv_idx}] Test"):
                try:
                    pred = method.question(qa["question"], qa.get("images", []))
                    f1, bleu, ref_used = self._get_best_metrics(pred, qa["refs"])

                    judge_label, judge_rationale = None, None
                    if self.judge and ref_used:
                        j = self.judge.score(qa["question"], ref_used, pred, images=qa.get("images", []))
                        judge_label, judge_rationale = j.get("label"), j.get("rationale")

                    cat_name = CATEGORIES.get(qa['category'], str(qa['category']))
                    raw_entry = {
                        "id": qa["qid"], "conversation_id": conv["cid"],
                        "question": qa["question"], "prediction": pred,
                        "reference_used": ref_used, "f1": f1, "bleu1": bleu,
                        "judge_label": judge_label, "judge_rationale": judge_rationale
                    }
                    local_results.setdefault(cat_name, []).append(raw_entry)
                except Exception as e:
                    print(f"Error in conversation {conv_idx}: {e}")

            done = sum(len(v) for v in local_results.values())
            conv_summary = self.calc_summary(local_results, done)
            method.over(**conv_summary)

            return conv_idx, local_results
        except Exception as e:
            print(f"Error in conversation {conv_idx}: {e}")
        finally:
            self.method_pool.put(method)

    def evaluate(self, conversations, max_samples=None, n_sample_conv=100, n_sample_msg=10000):
        all_conv_results = {}
        num_workers = self.method_pool.qsize()
        targets = conversations[:n_sample_conv]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(self._worker, i, conv, max_samples, n_sample_msg)
                for i, conv in enumerate(targets)
            ]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Total Progress"):
                idx, res = future.result()
                all_conv_results[idx] = res

        return self._aggregate_and_print(all_conv_results)

    def _aggregate_and_print(self, all_conv_results: dict) -> dict:
        merged_results = {}
        for local_results in all_conv_results.values():
            for cat, items in local_results.items():
                merged_results.setdefault(cat, []).extend(items)

        total_done = sum(len(items) for items in merged_results.values())
        final_data = self.calc_summary(merged_results, total_done)
        self.print_summary(final_data['summary'])
        return final_data

    def print_summary(self, summary: Summary) -> None:
        line = "=" * 60
        print(f"\n{line}\n EVALUATION SUMMARY \n{line}")
        print("Overall:")
        for metric, score in summary['overall'].items():
            print(f"  {metric:10s}: {score:.4f}")
        print("\nBy Category:")
        for cat, data in summary['by_category'].items():
            print(f"  {cat} (n={data['count']}):")
            for metric, score in data['metrics'].items():
                print(f"    {metric}: {score:.4f}")
        print(f"{line}\n")

    def evaluate_file(self, path, **kwargs):
        full_path = f"{self.database_root_path}/{path}"
        with open(full_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f] if path.endswith(".jsonl") else json.load(f)
        conversations = self._collect_conversations(data)
        return self.evaluate(conversations, **kwargs)