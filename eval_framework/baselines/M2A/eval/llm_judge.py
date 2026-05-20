import json

JUDGE_SYSTEM = (
"""
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.
You will be given the following data:
(1) a question (posed by one user to another user),
(2) a 'gold' (ground truth) answer,
(3) a generated answer,
which you will score as CORRECT or WRONG.
The point of the question is to ask about something one user should know about the other user based on their
prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be longer, but you should be generous with your grading — as long as it touches
on the same topic as the gold answer, it should be counted as CORRECT.
For time-related questions, the gold answer will be a specific date, month, or year. The generated answer
might include relative references (e.g., "last Tuesday"), but you should be generous — if it refers to
the same time period as the gold answer, mark it CORRECT, even if the format differs (e.g., "May 7th" vs.
"7 May").

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.
Return the label in JSON format with the key as "label".


Output order:
One-sentence rationale explaining your decision (do not use the words “CORRECT” or “WRONG”).
On the next line,œ output the final label exactly as: CORRECT or WRONG.
Then output JSON: {\"label\":\"CORRECT\"} or {\"label\":\"WRONG\"}.

Now it's time for the real question:
"""
)

def _img_from(x):
    return x

def _norm_imgs(imgs):
    if imgs is None:
        return []
    if isinstance(imgs, (str, dict)):
        imgs = [imgs]
    return [ _img_from(v) for v in imgs ]

class LLMJudge:
    def __init__(self, base_url, api_key, model=None):
        from langchain_openai import ChatOpenAI
        from openai import OpenAI

        if model is None:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )
            models = client.models.list()
            model = models.data[0].id

        self.llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            api_key=api_key,
        )
        
    def score(self, question, gold, pred, images=None):
        imgs = _norm_imgs(images)

        user_text = (
            f"Question:\n{question}\n\n"
            f"Gold answer:\n{gold}\n\n"
            f"Generated answer:\n{pred}"
        )

        content = []
        for img in imgs:
            img = "file://" + img
            content.append({
                "type": "image_url",
                "image_url": {"url": img}
            })
        content.append({
            "type": "text",
            "text": user_text
        })

        messages = [
            {
                "role": "system",
                "content": JUDGE_SYSTEM
            },
            {
                "role": "user",
                "content": content
            }
        ]

        resp = self.llm.invoke(messages)
        text = resp.content.strip()

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        label = None
        rationale = ""

        for i, l in enumerate(lines):
            if l in ("CORRECT", "WRONG"):
                label = l
                if i > 0:
                    rationale = lines[0]
                break

        if label is None:
            try:
                jstart = text.rfind("{")
                if jstart != -1:
                    j = json.loads(text[jstart:])
                    v = str(j.get("label", "")).upper()
                    if v in ("CORRECT", "WRONG"):
                        label = v
            except:
                pass

        if label not in ("CORRECT", "WRONG"):
            label = "WRONG"

        if rationale == "" and lines:
            if lines[0] not in ("CORRECT", "WRONG"):
                rationale = lines[0]

        return {
            "rationale": rationale,
            "label": label,
            "raw": text
        }

