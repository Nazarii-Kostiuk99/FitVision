"""
Qualitative feedbakc generation usingGroq API.

After main pipeline produces per-rep metrics, they are sent to a
small language model hosted on Groq and gets back small coash-like
qualitative feedback
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_client = None


def _get_client():
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set")
        _client = Groq(api_key=api_key)
    return _client


def _build_set_prompt(exercise_type, rep_feedback_blocks):
    # build a summary of all reps + pass to the LLM
    reps_str = ""
    for block in rep_feedback_blocks:
        reps_str += f"\nRep {block['rep']}:\n"
        for line in block["feedback"]:
            reps_str += f"  - {line}\n"

    prompt = (
        f"A user just completed a set of {exercise_type}s. "
        f"Here are the automatically measured results for each rep:\n"
        f"{reps_str}\n"
        f"Give a short overall summary of the set (2-3 sentences). "
        f"Mention the specific rep numbers where issues occurred and what the issue was. "
        f"If the same issue appears in multiple reps, group them together (e.g. 'reps 3, 5 and 7'). "
        f"End with a sentence on the overall quality of the set. "
        f"Be encouraging but honest. No jargon. Do not repeat scores or numbers from the metrics."
    )
    return prompt


# takes all rep blocks + generate overall summary
def generate_set_feedback(
    exercise_type, rep_feedback_blocks, model="llama-3.1-8b-instant"
):
    try:
        client = _get_client()
        prompt = _build_set_prompt(exercise_type, rep_feedback_blocks)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful fitness coach giving brief, practical feedback "
                        "on exercise form. Be encouraging but honest."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=250,  
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"(LLM feedback unavailable: {e})"
