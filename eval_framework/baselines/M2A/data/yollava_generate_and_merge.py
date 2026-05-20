import os
import re
import json
import glob
import time
import base64
import random
import argparse
import hashlib
import logging
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Optional, DefaultDict
from collections import defaultdict
from datetime import datetime, timedelta

import httpx
from tqdm import tqdm
from dateutil import parser as dtparser
from openai import OpenAI, BadRequestError, APIConnectionError, APITimeoutError, RateLimitError

# ==============================================================================
# Logging Setup
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ==============================================================================
# Constants & Regex
# ==============================================================================
_DIA_SD_RE = re.compile(r"[SD](\d+):(\d+)")
_DIA_S_STRICT_RE = re.compile(r"^S(\d+):(\d+)$")

DEFAULT_OPENAI_MODEL = "gemini-3-pro-preview"
DEFAULT_TIMEOUT = 250.0

# ==============================================================================
# Utility Functions
# ==============================================================================

def make_openai_client(base_url: str, api_key: str) -> OpenAI:
    """Initializes the OpenAI client with custom headers and timeout settings."""
    if not api_key:
        raise ValueError("API Key is missing. Please provide it via args or env 'OPENAI_API_KEY'.")
    
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        default_headers={
            "x-api-key": api_key,
            "api-key": api_key,
        },
        http_client=httpx.Client(timeout=DEFAULT_TIMEOUT),
        max_retries=0,  # Manual retry logic is implemented in the caller
    )

def ensure_dir(path: str):
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def encode_image_to_data_url(path: str) -> str:
    """Encodes an image file to a base64 data URL string."""
    ext = os.path.splitext(path)[1].lower().strip(".")
    mime = "image/png" if ext == "png" else ("image/webp" if ext == "webp" else "image/jpeg")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def parse_time(s: str) -> datetime:
    return dtparser.parse(s, fuzzy=True)

def format_time(dt: datetime) -> str:
    # Mimics format: "3:12 pm on 5 June, 2023"
    hour = dt.strftime("%I").lstrip("0") or "0"
    minute = dt.strftime("%M")
    ampm = dt.strftime("%p").lower()
    day = str(int(dt.strftime("%d")))
    month = dt.strftime("%B")
    year = dt.strftime("%Y")
    return f"{hour}:{minute} {ampm} on {day} {month}, {year}"

def generate_intermediate_times(start_str: str, end_str: str, n: int) -> List[str]:
    """Generates `n` unique, strictly increasing timestamps between start and end."""
    if n < 1:
        raise ValueError("n must be >= 1")
    
    left = parse_time(start_str)
    right = parse_time(end_str)
    
    if left >= right:
        raise ValueError(f"Time order violated: start ({start_str}) >= end ({end_str})")

    delta = right - left
    out = []
    for i in range(n):
        frac = (i + 1) / (n + 1)
        ti = left + frac * delta
        # Boundary safety
        if ti <= left: ti = left + timedelta(seconds=1)
        if ti >= right: ti = right - timedelta(seconds=1)
        out.append(format_time(ti))
    
    # Validation
    if len(set(out)) != len(out):
        raise ValueError("Generated times are not unique.")
        
    return out

def slugify_caption(s: str, max_words: int = 10, max_len: int = 80) -> str:
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = "_".join(s.split(" ")[:max_words])
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s[:max_len] if s else "caption"

def get_image_filename_from_caption(caption: str) -> str:
    slug = slugify_caption(caption)
    h12 = hashlib.sha1(caption.encode("utf-8")).hexdigest()[:12]
    return f"{slug}_{h12}.jpg"

def match_image_by_caption(caption: str, dataset_root: str, image_dir: str) -> Optional[str]:
    """Attempts to find the image file corresponding to a generated caption."""
    if not isinstance(caption, str) or not caption.strip():
        return None
        
    fname = get_image_filename_from_caption(caption)
    abs_path = os.path.join(image_dir, fname)
    
    if not os.path.isfile(abs_path):
        return None
        
    return os.path.relpath(abs_path, start=dataset_root)

def compute_qa_distribution(min_total: int = 15) -> Tuple[int, Dict[int, int]]:
    """
    Enforces category ratio 6:7:8:9 = 2:3:1:4.
    Total count will be a multiple of 10.
    """
    if min_total < 1:
        raise ValueError("min_total must be >= 1")
        
    k = (min_total + 9) // 10
    if k < 2: k = 2  # Ensure minimal viable size
    
    total = 10 * k
    counts = {6: 2 * k, 7: 3 * k, 8: 1 * k, 9: 4 * k}
    return total, counts

# ==============================================================================
# Data Normalization (LoCoMo -> YoLLaVA)
# ==============================================================================

def normalize_locomo_sample(
    sample: Dict[str, Any], 
    sample_idx: int, 
    dataset_root: str, 
    image_dir: str
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Converts a single LoCoMo sample format into the YoLLaVA schema.
    Returns the normalized sample and a list of QA pairs missing evidence.
    """
    conv = sample.get("conversation", {})
    spk_a = conv.get("speaker_a")
    spk_b = conv.get("speaker_b")
    
    # Extract session indices
    old_si_list = sorted([
        int(k.split("_")[1]) for k in conv.keys() 
        if k.startswith("session_") and k.count("_") == 1
    ])
    
    if not old_si_list:
        raise ValueError("No sessions found in conversation.")

    new_conv = {"speaker_0": spk_a, "speaker_1": spk_b, "n_session": len(old_si_list)}
    old_to_new_si = {old: new for new, old in enumerate(old_si_list)}
    old_session_len = {}

    # Normalize Sessions
    for new_si, old_si in enumerate(old_si_list):
        key_sess = f"session_{old_si}"
        key_time = f"session_{old_si}_date_time"
        
        raw_sess = conv.get(key_sess, [])
        old_session_len[old_si] = len(raw_sess)
        
        new_sess = []
        for ui, turn in enumerate(raw_sess):
            speaker_name = turn.get("speaker")
            text = turn.get("text")
            
            # Map speaker name to ID (0 or 1)
            if speaker_name == spk_a:
                spk_id = 0
            elif speaker_name == spk_b:
                spk_id = 1
            else:
                raise ValueError(f"Unknown speaker: {speaker_name}")

            norm_turn = {
                "speaker": spk_id,
                "dia_id": f"S{new_si}:{ui}",
                "images": [],
                "text": text
            }

            # Handle Image Matching
            img_url = turn.get("img_url")
            if img_url:
                rel_img = match_image_by_caption(turn.get("blip_caption"), dataset_root, image_dir)
                if rel_img:
                    norm_turn["images"] = [rel_img]

            new_sess.append(norm_turn)

        new_conv[f"session_{new_si}"] = new_sess
        new_conv[f"session_{new_si}_date_time"] = conv.get(key_time)

    # Normalize QA
    new_qa_list = []
    missing_evidence_report = []

    for qi, qa in enumerate(sample.get("qa", [])):
        q_text = str(qa.get("question"))
        answer = qa.get("answer")
        
        # Serialize answer if complex type
        if not isinstance(answer, (str, int, float, bool)):
            answer = json.dumps(answer, ensure_ascii=False)

        # Map evidence IDs
        ev_new = []
        raw_evidence = qa.get("evidence", [])
        
        # Extract all dia_id references (e.g., S1:5)
        tokens = []
        for item in raw_evidence:
            tokens.extend([m.group(0) for m in _DIA_SD_RE.finditer(item)])

        for tok in tokens:
            m = _DIA_SD_RE.search(tok)
            old_si, old_ui_1based = int(m.group(1)), int(m.group(2))
            
            if old_si in old_to_new_si:
                ui = old_ui_1based - 1  # Convert to 0-based
                if 0 <= ui < old_session_len[old_si]:
                    new_si = old_to_new_si[old_si]
                    ev_new.append(f"S{new_si}:{ui}")

        if not ev_new:
            missing_evidence_report.append({
                "sample_index": sample_idx, 
                "qa_index": qi, 
                "question": q_text
            })

        qa_obj = {
            "question": {"text": q_text, "image": []},
            "answer": answer,
            "evidence": ev_new
        }
        if "category" in qa:
            qa_obj["category"] = qa["category"]
            
        new_qa_list.append(qa_obj)

    return {"conversation": new_conv, "qa": new_qa_list}, missing_evidence_report

# ==============================================================================
# Prompt Construction & LLM Interaction
# ==============================================================================

def build_multimodal_payload(
    prompt_text: str, 
    concept_images_rel: Dict[str, List[str]], 
    dataset_root: str, 
    max_images: int
) -> List[Dict[str, Any]]:
    """Builds the message payload for OpenAI Vision API."""
    content = [{"type": "text", "text": prompt_text}]

    # Flatten image list
    valid_images = []
    if max_images > 0:
        # Round-robin selection from concepts to ensure diversity
        concept_queues = {k: v[:] for k, v in concept_images_rel.items() if v}
        keys = list(concept_queues.keys())
        
        while len(valid_images) < max_images and any(concept_queues.values()):
            for k in keys:
                if concept_queues[k]:
                    rel_path = concept_queues[k].pop(0)
                    abs_path = os.path.join(dataset_root, rel_path)
                    if os.path.exists(abs_path):
                        valid_images.append((rel_path, abs_path))
                    if len(valid_images) >= max_images:
                        break

    if not valid_images:
        raise ValueError("No valid images found for multimodal prompt.")

    # Add mapping instructions
    mapping_str = "\n".join([f"IMG#{i} -> {rel}" for i, (rel, _) in enumerate(valid_images)])
    content.append({"type": "text", "text": f"IMAGE_TO_PATH_MAPPING:\n{mapping_str}"})

    # Add image data
    for i, (rel, abs_path) in enumerate(valid_images):
        data_url = encode_image_to_data_url(abs_path)
        content.append({"type": "text", "text": f"IMG#{i} (path: '{rel}'):"})
        content.append({"type": "image_url", "image_url": {"url": data_url}})

    return content

def build_generation_prompt(
    concept_names: List[str],
    concept_images_rel: Dict[str, List[str]],
    speakers: Tuple[str, str],
    turns_range: Tuple[int, int],
    fixed_session_times: List[str],
    qa_range: Tuple[int, int]
) -> str:
    """Constructs the system prompt for generating conversation and QA."""
    q_total, q_counts = compute_qa_distribution(min_total=max(20, qa_range[0]))
    spk0, spk1 = speakers

    # Define the template
    lines = [
        "You are an expert dataset generator. Output ONE JSON object with keys: whitelist, conversation, qa_indexed.",
        "",
        f"CONCEPT_GROUP (embed these in angle brackets <Name>): {json.dumps(concept_names)}",
        f"ALLOWED_IMAGE_PATHS: {json.dumps(concept_images_rel, indent=2)}",
        f"SPEAKERS: {json.dumps({'speaker_0': spk0, 'speaker_1': spk1})}",
        "",
        "REQUIREMENTS:",
        f"1. Conversation: {len(fixed_session_times)} sessions. Each session {turns_range[0]}-{turns_range[1]} turns.",
        "2. Format: {speaker: 0|1, dia_id: 'S{sess}:{turn}', images: [], text: '...'}.",
        "3. Whitelist: Ordered list of all produced dia_ids.",
        "4. FIRST MSG: Must contain ALL concept names in angle brackets (e.g., 'I saw <cat> and <dog>').",
        "5. Images: Only use allowed paths. Ground text in visual details.",
        "",
        f"FIXED TIMESTAMPS (Strictly adhere): {json.dumps(fixed_session_times)}",
        "",
        f"QA GENERATION ({q_total} total):",
        " - Category 6 (Multi-Hop), 7 (Temporal), 8 (Open), 9 (Single-Hop).",
        f" - RATIO MUST BE EXACT: {q_counts[6]}xCat6, {q_counts[7]}xCat7, {q_counts[8]}xCat8, {q_counts[9]}xCat9.",
        " - evidence_index: List of integers mapping to whitelist indices.",
        "",
        "STRICT JSON OUTPUT ONLY."
    ]
    return "\n".join(lines)

def query_openai_with_retry(
    client: OpenAI,
    model: str,
    prompt_text: str,
    concept_images_rel: Dict[str, List[str]],
    dataset_root: str,
    max_images: int,
    temperature: float,
    max_attempts: int = 5,
    reinit_args: Tuple[str, str] = None
) -> Tuple[Optional[Dict], str]:
    """Handles OpenAI API calls with exponential backoff and client rebuilding."""
    
    parts = build_multimodal_payload(prompt_text, concept_images_rel, dataset_root, max_images)
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Output STRICT JSON."},
                    {"role": "user", "content": parts}
                ],
            )
            content = resp.choices[0].message.content
            if not content:
                raise ValueError("Empty response content.")
            
            return json.loads(content), content

        except Exception as e:
            last_error = e
            msg = str(e).lower()
            is_auth_error = "api key" in msg or "unauthorized" in msg
            
            if attempt == max_attempts:
                break
            
            sleep_s = min(2 ** (attempt - 1), 16)
            logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {sleep_s}s...")
            time.sleep(sleep_s)

            # Rebuild client if it seems like an auth/connection issue
            if is_auth_error and reinit_args:
                try:
                    client = make_openai_client(*reinit_args)
                except Exception as build_err:
                    logger.error(f"Failed to rebuild client: {build_err}")

    return None, str(last_error)

# ==============================================================================
# Validation Logic
# ==============================================================================

def validate_generated_content(data: Dict[str, Any], fixed_times: List[str], concepts: List[str]):
    """Performs strict schema and logic checks on the generated JSON."""
    if not isinstance(data, dict):
        raise ValueError("Root is not a dict")
    
    conv = data.get("conversation")
    whitelist = data.get("whitelist")
    qa_idx = data.get("qa_indexed")

    # 1. Conversation Structure
    if not conv or "n_session" not in conv:
        raise ValueError("Missing conversation or n_session")
    
    n_sess = conv["n_session"]
    if n_sess != len(fixed_times):
        raise ValueError(f"Session count mismatch: got {n_sess}, expected {len(fixed_times)}")

    generated_ids = []
    
    # 2. Session and Time Validation
    for i, expected_time in enumerate(fixed_times):
        time_key = f"session_{i}_date_time"
        sess_key = f"session_{i}"
        
        if conv.get(time_key) != expected_time:
            raise ValueError(f"Time mismatch at {i}: expected {expected_time}")
            
        sess_msgs = conv.get(sess_key, [])
        if not sess_msgs:
            raise ValueError(f"Session {i} is empty")
            
        for u, msg in enumerate(sess_msgs):
            expected_id = f"S{i}:{u}"
            if msg.get("dia_id") != expected_id:
                raise ValueError(f"ID mismatch: got {msg.get('dia_id')}, expected {expected_id}")
            generated_ids.append(expected_id)

    # 3. Whitelist Validation
    if whitelist != generated_ids:
        raise ValueError("Whitelist does not match generated dialogue IDs order.")

    # 4. Concept inclusion check (First message)
    first_msg = conv["session_0"][0]["text"]
    for c in concepts:
        if f"<{c}>" not in first_msg:
            raise ValueError(f"Concept <{c}> missing from first message: {first_msg[:50]}...")

    # 5. QA Validation
    if not qa_idx:
        raise ValueError("QA list is empty")
        
    for item in qa_idx:
        ev = item.get("evidence_index")
        cat = item.get("category")
        if not ev or not isinstance(ev, list):
            raise ValueError("Invalid evidence_index")
        if any(idx >= len(whitelist) for idx in ev):
            raise ValueError("Evidence index out of bounds")
        if cat not in (6, 7, 8, 9):
            raise ValueError(f"Invalid category: {cat}")

# ==============================================================================
# Merge & Injection Logic
# ==============================================================================

def merge_generated_into_sample(
    sample: Dict[str, Any], 
    generated_group: Dict[str, Any], 
    insert_pos: int
):
    """
    Merges a generated conversation group into the host sample at a specific position.
    Remaps all dialogue IDs (both host and inserted) and updates QA evidence.
    """
    host_conv = sample["conversation"]
    gen_conv = generated_group["conversation"]
    
    # Extract Host Sessions
    host_sessions = []
    host_times = []
    n_host = host_conv["n_session"]
    for i in range(n_host):
        host_sessions.append(deepcopy(host_conv[f"session_{i}"]))
        host_times.append(host_conv[f"session_{i}_date_time"])

    # Extract Generated Sessions
    gen_sessions = []
    gen_times = []
    n_gen = gen_conv["n_session"]
    for i in range(n_gen):
        gen_sessions.append(deepcopy(gen_conv[f"session_{i}"]))
        gen_times.append(gen_conv[f"session_{i}_date_time"])

    # Merge Lists
    merged_sessions = host_sessions[:insert_pos] + gen_sessions + host_sessions[insert_pos:]
    merged_times = host_times[:insert_pos] + gen_times + host_times[insert_pos:]
    
    # Track origin for ID remapping: ("host", old_si) or ("gen", old_si)
    origin_map = (
        [("host", i) for i in range(insert_pos)] + 
        [("gen", i) for i in range(n_gen)] + 
        [("host", i) for i in range(insert_pos, n_host)]
    )

    # Reindex and build mapping tables
    new_sessions_data = []
    host_id_map = {} # (old_si, ui) -> new_dia_id
    gen_id_map = {}  # (old_si, ui) -> new_dia_id

    for new_si, (source_type, old_si) in enumerate(origin_map):
        sess_data = merged_sessions[new_si]
        new_sess_objs = []
        for ui, turn in enumerate(sess_data):
            turn_copy = deepcopy(turn)
            new_dia_id = f"S{new_si}:{ui}"
            turn_copy["dia_id"] = new_dia_id
            new_sess_objs.append(turn_copy)
            
            if source_type == "host":
                host_id_map[(old_si, ui)] = new_dia_id
            else:
                gen_id_map[(old_si, ui)] = new_dia_id
        new_sessions_data.append(new_sess_objs)

    # Update Host QA Evidence
    for qa in sample["qa"]:
        if "evidence" in qa:
            new_ev = []
            for tok in qa["evidence"]:
                m = _DIA_S_STRICT_RE.match(tok)
                if m:
                    key = (int(m.group(1)), int(m.group(2)))
                    if key in host_id_map:
                        new_ev.append(host_id_map[key])
            qa["evidence"] = new_ev

    # Append Generated QA with remapped evidence
    for qa_item in generated_group["qa"]:
        new_ev = []
        for tok in qa_item["evidence"]:
            m = _DIA_S_STRICT_RE.match(tok)
            if m:
                key = (int(m.group(1)), int(m.group(2)))
                if key in gen_id_map:
                    new_ev.append(gen_id_map[key])
        
        # Add to sample QA list
        qa_copy = deepcopy(qa_item)
        qa_copy["evidence"] = new_ev
        sample["qa"].append(qa_copy)

    # Rebuild Conversation Object
    host_conv.clear()
    host_conv["speaker_0"] = sample["conversation"].get("speaker_0", "A") # Fallback safety
    host_conv["speaker_1"] = sample["conversation"].get("speaker_1", "B")
    host_conv["n_session"] = len(new_sessions_data)
    
    for i, (sess, tm) in enumerate(zip(new_sessions_data, merged_times)):
        host_conv[f"session_{i}"] = sess
        host_conv[f"session_{i}_date_time"] = tm

def inject_vqa_data(sample: Dict[str, Any], vqa_db: Dict[str, List[Dict]]):
    """Scans conversation images and injects pre-defined VQA pairs (Category 0)."""
    conv = sample["conversation"]
    
    # Map concepts to dialogue IDs where they appear
    concept_occurrences = defaultdict(list)
    n_sess = conv.get("n_session", 0)
    
    for si in range(n_sess):
        sess = conv.get(f"session_{si}", [])
        for ui, turn in enumerate(sess):
            for img_path in turn.get("images", []):
                # Assume concept is the parent folder name
                concept = os.path.basename(os.path.dirname(img_path)).lower()
                concept_occurrences[concept].append(f"S{si}:{ui}")

    # Inject QA
    for concept, dia_ids in concept_occurrences.items():
        vqa_items = vqa_db.get(concept, [])
        unique_dia_ids = sorted(list(set(dia_ids)))
        
        for item in vqa_items:
            ans = item.get("answer_text")
            if not isinstance(ans, (str, int, float, bool)):
                ans = json.dumps(ans, ensure_ascii=False)

            sample["qa"].append({
                "question": {"text": item["question_text"], "image": [item["image_path"]]},
                "answer": ans,
                "evidence": unique_dia_ids,
                "category": 5  # 0 indicates external VQA injection
            })

# ==============================================================================
# Main Execution
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="YoLLaVA Dataset Generator & Merger")
    
    # Paths
    parser.add_argument("--dataset_root", type=str, required=True, help="Root directory for absolute path resolution")
    parser.add_argument("--locomo_json", type=str, required=True, help="Input LoCoMo JSON file")
    parser.add_argument("--yollava_train_dir", type=str, required=True, help="Directory containing concept image folders")
    parser.add_argument("--image_dir", type=str, required=True, help="Base directory for image lookup")
    parser.add_argument("--vqa_json", type=str, required=True, help="Source JSON for VQA injection")
    
    # Output
    parser.add_argument("--out_path", type=str, default="output/merged.json", help="Final merged JSON path")
    parser.add_argument("--intermediate_dir", type=str, default="output/intermediate", help="Directory for intermediate generations")
    
    # Generation Parameters
    parser.add_argument("--min_group_size", type=int, default=3)
    parser.add_argument("--max_group_size", type=int, default=4)
    parser.add_argument("--img_per_concept", type=int, nargs=2, default=[2, 3], help="Min Max images per concept")
    parser.add_argument("--sessions", type=int, nargs=2, default=[5, 6], help="Min Max sessions per group")
    parser.add_argument("--turns", type=int, nargs=2, default=[5, 15], help="Min Max turns per session")
    parser.add_argument("--qa_count", type=int, nargs=2, default=[20, 20], help="Min Max QA pairs per group")
    
    # API & Model
    parser.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY"), help="OpenAI API Key")
    parser.add_argument("--api_base", type=str, default="https://api.openai.com/v1", help="OpenAI Base URL")
    parser.add_argument("--model", type=str, default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_images_request", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    
    # Init
    random.seed(args.seed)
    ensure_dir(os.path.dirname(args.out_path))
    ensure_dir(args.intermediate_dir)
    
    # Load Data
    logger.info("Loading LoCoMo dataset...")
    with open(args.locomo_json, "r", encoding="utf-8") as f:
        locomo_raw = json.load(f)
        
    logger.info("Normalizing LoCoMo data...")
    dataset, _ = [], []
    for idx, sample in enumerate(tqdm(locomo_raw, desc="Normalizing")):
        norm_sample, _ = normalize_locomo_sample(sample, idx, args.dataset_root, args.image_dir)
        dataset.append(norm_sample)

    # Prepare Concepts
    concepts = [d for d in os.listdir(args.yollava_train_dir) 
                if os.path.isdir(os.path.join(args.yollava_train_dir, d)) and not d.startswith(".")]
    if not concepts:
        raise ValueError("No concept folders found in train dir.")

    # Prepare API Client
    client = make_openai_client(args.api_base, args.api_key)

    # --------------------------------------------------------------------------
    # Phase 1: Generation
    # --------------------------------------------------------------------------
    generated_data = [] # Stores (sample_idx, insert_pos, content)
    
    logger.info("Starting Generation Phase...")
    for si, sample in enumerate(tqdm(dataset, desc="Generating")):
        conv = sample["conversation"]
        n_host = conv["n_session"]
        
        if n_host < 2:
            logger.debug(f"Skipping sample {si}: too few sessions ({n_host})")
            continue

        # Determine insertion timing
        insert_pos = random.randint(1, n_host - 1)
        t_left = conv[f"session_{insert_pos-1}_date_time"]
        t_right = conv[f"session_{insert_pos}_date_time"]
        
        # Determine Generation Params
        n_gen_sessions = random.randint(args.sessions[0], args.sessions[1])
        try:
            fixed_times = generate_intermediate_times(t_left, t_right, n_gen_sessions)
        except ValueError as e:
            logger.warning(f"Time generation failed for sample {si}: {e}. Skipping.")
            continue

        # Select Concepts & Images
        group_concepts = random.sample(concepts, min(len(concepts), random.randint(args.min_group_size, args.max_group_size)))
        concept_imgs_rel = {}
        for c in group_concepts:
            c_dir = os.path.join(args.yollava_train_dir, c)
            all_imgs = sorted(glob.glob(os.path.join(c_dir, "*")))
            k = min(len(all_imgs), random.randint(args.img_per_concept[0], args.img_per_concept[1]))
            concept_imgs_rel[c] = [os.path.relpath(p, args.dataset_root) for p in random.sample(all_imgs, k)]

        # Construct Prompt & Call API
        prompt = build_generation_prompt(
            group_concepts, concept_imgs_rel, 
            (conv["speaker_0"], conv["speaker_1"]),
            args.turns, fixed_times, args.qa_count
        )

        resp_obj, raw_resp = query_openai_with_retry(
            client=client,
            model=args.model,
            prompt_text=prompt,
            concept_images_rel=concept_imgs_rel,
            dataset_root=args.dataset_root,
            max_images=args.max_images_request,
            temperature=args.temperature,
            reinit_args=(args.api_base, args.api_key)
        )

        if resp_obj:
            try:
                # Post-process response to match internal format
                # The model returns a flat 'qa_indexed', we convert to 'qa' list here
                whitelist = resp_obj["whitelist"]
                qa_final = []
                for item in resp_obj["qa_indexed"]:
                    qa_final.append({
                        "question": {"text": item["question"], "image": []},
                        "answer": item["answer"],
                        "evidence": [whitelist[i] for i in item["evidence_index"]],
                        "category": item["category"]
                    })
                
                final_group = {
                    "conversation": resp_obj["conversation"],
                    "qa": qa_final
                }
                
                validate_generated_content(resp_obj, fixed_times, group_concepts)
                
                # Save artifact
                dump_path = os.path.join(args.intermediate_dir, f"gen_{si:04d}.json")
                with open(dump_path, "w") as f:
                    json.dump(final_group, f, indent=2)

                generated_data.append((si, insert_pos, final_group))
                
            except ValueError as ve:
                logger.error(f"Validation failed for sample {si}: {ve}")
        else:
            logger.error(f"Generation failed completely for sample {si}. Error: {raw_resp[:200]}")

    # --------------------------------------------------------------------------
    # Phase 2: Merging
    # --------------------------------------------------------------------------
    logger.info("Merging generated content...")
    
    # Index generated content by sample ID
    gen_map = defaultdict(list)
    for si, pos, content in generated_data:
        gen_map[si].append((pos, content))

    for si, sample in enumerate(tqdm(dataset, desc="Merging")):
        if si in gen_map:
            # Sort insertions by position (though logic allows 1 per sample currently)
            inserts = sorted(gen_map[si], key=lambda x: x[0])
            # Process strictly one insertion as per requirements
            pos, content = inserts[0]
            merge_generated_into_sample(sample, content, pos)

    # --------------------------------------------------------------------------
    # Phase 3: VQA Injection
    # --------------------------------------------------------------------------
    logger.info("Injecting external VQA data...")
    
    # Load VQA DB
    with open(args.vqa_json, "r") as f:
        vqa_raw = json.load(f)
    
    # Normalize VQA DB structure
    vqa_db = defaultdict(list)
    for concept, items in vqa_raw.items():
        for img_rel, data in items.items():
            # Extract standard Answer
            ans = data.get("answer", "")
            if "options" in data and "correct_answer" in data:
                ans = data["options"][data["correct_answer"]]
            
            vqa_db[concept.lower()].append({
                "image_path": img_rel,
                "question_text": data.get("question", ""),
                "answer_text": ans
            })

    for sample in tqdm(dataset, desc="Injecting VQA"):
        inject_vqa_data(sample, vqa_db)

    # --------------------------------------------------------------------------
    # Save
    # --------------------------------------------------------------------------
    logger.info(f"Saving final dataset to {args.out_path}...")
    with open(args.out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    logger.info("Done.")

if __name__ == "__main__":
    main()