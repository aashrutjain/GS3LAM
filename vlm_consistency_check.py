"""
VLM safety-score consistency check (GS3LAM_PAPER_SCOPE.md, "This Summer's Remaining
Experimental Work", item 1).

Standalone: does not import anything from src/ or touch src/cbf/, does not need a GPU,
GS3LAM training, or real splats. Queries the exact safety-auditor prompt from
vlm_safety_score.py's query_vlm_safety() against Gemini 5 times per object, at the
same call configuration already used in production (temperature=0.2, which already
satisfies "temperature > 0"), to test whether the score is stable or noisy.

Images are stock photos (assets/vlm_consistency/images/), not real Replica hero-frame
crops — no data/Replica exists in this checkout. See GS3LAM_PAPER_SCOPE.md for that
caveat and the Gap Tracking note about redoing this with real data in the fall.
"""

import glob
import json
import os
import statistics

from dotenv import load_dotenv
from google import genai
from google.genai import types
import PIL.Image

load_dotenv()

GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=GEMINI_API_KEY)

IMAGES_DIR = "assets/vlm_consistency/images"
RESULTS_PATH = "assets/vlm_consistency/results.json"
QUERIES_PER_OBJECT = 5

# Verbatim from vlm_safety_score.py:query_vlm_safety() — do not edit here.
PROMPT = """
    You are a physical safety auditor for a 3kg wheeled robot (TurtleBot4).
    Analyze the physical materials, structure, and stability of the isolated object in this image.
    Output a strictly formatted JSON dictionary with a single key 'safety_score', holding a float from 0.0 (lethal hazard/fragile/easily tipped/cables) to 1.0 (completely safe to drive on/flat solid ground).
    """
TEMPERATURE = 0.2  # matches vlm_safety_score.py's existing config; already > 0.

# vlm_safety_score.py hardcodes model='gemini-1.5-flash', but as of this session
# (2026-07-09) that model has been fully deprecated and 404s for every call under
# this API key (confirmed via client.models.list()) — this is a real production bug
# in the existing pipeline, independent of anything this script tests. 'gemini-2.5-flash'
# was tried next and is mid-sunset (intermittent 404 "no longer available" mid-run).
# Using the rolling 'gemini-flash-latest' alias instead so the prompt itself can still
# be tested; this substitution is flagged in the results/report, not silent.
MODEL = "gemini-flash-latest"
MAX_RETRIES = 2

# 'gemini-flash-latest' spends output-token budget on an internal "thinking" pass
# before the visible answer; with no max_output_tokens set, the JSON answer was
# getting cut off mid-value (e.g. '{"safety_score": 0.1' with no closing brace) on a
# large fraction of calls even though finish_reason reported STOP. gemini-1.5-flash
# (the original pinned model) predates "thinking" models and never needed this.
# Disabling the thinking budget and raising max_output_tokens fixes it; this is an
# infra/config adaptation to make the unchanged prompt work on a current model, not a
# change to the prompt or the safety-score scale itself.
GENERATION_CONFIG = types.GenerateContentConfig(
    response_mime_type="application/json",
    temperature=TEMPERATURE,
    max_output_tokens=1024,
    thinking_config=types.ThinkingConfig(thinking_budget=0),
)


def query_once(img: PIL.Image.Image) -> float:
    response = client.models.generate_content(
        model=MODEL,
        contents=[PROMPT, img],
        config=GENERATION_CONFIG,
    )
    result = json.loads(response.text)
    return float(result["safety_score"])


def main():
    image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGES_DIR}")

    all_results = {}

    for path in image_paths:
        obj_name = os.path.splitext(os.path.basename(path))[0]
        img = PIL.Image.open(path)
        print(f"\nQuerying {obj_name} x{QUERIES_PER_OBJECT}...")

        scores = []
        for i in range(QUERIES_PER_OBJECT):
            for attempt in range(MAX_RETRIES + 1):
                try:
                    score = query_once(img)
                    scores.append(score)
                    print(f"  [{i+1}/{QUERIES_PER_OBJECT}] safety_score = {score:.3f}")
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        print(f"  [{i+1}/{QUERIES_PER_OBJECT}] retry {attempt+1} after error: {e}")
                    else:
                        print(f"  [{i+1}/{QUERIES_PER_OBJECT}] ERROR (out of retries): {e}")

        all_results[obj_name] = {
            "scores": scores,
            "mean": statistics.mean(scores) if scores else None,
            "stdev": statistics.stdev(scores) if len(scores) > 1 else None,
        }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 88)
    print(f"{'Object':30s} {'Scores':32s} {'Mean':>8s} {'StdDev':>8s}")
    print("-" * 88)
    for obj_name, r in all_results.items():
        scores_str = ", ".join(f"{s:.2f}" for s in r["scores"])
        mean_str = f"{r['mean']:.3f}" if r["mean"] is not None else "N/A"
        std_str = f"{r['stdev']:.3f}" if r["stdev"] is not None else "N/A"
        print(f"{obj_name:30s} {scores_str:32s} {mean_str:>8s} {std_str:>8s}")
    print("=" * 88)
    print(f"\nRaw results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
