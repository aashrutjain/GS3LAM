"""
VLM safety-score consistency check, real-data version (GS3LAM_PAPER_SCOPE.md, "This
Summer's Remaining Experimental Work", item 1 -- real-Replica-data follow-up to the
2026-07-09 stock-photo run).

Standalone: does not import anything from src/ or touch src/cbf/, does not need a GPU,
GS3LAM training, or real splats/classifier. Queries the exact safety-auditor prompt
from vlm_safety_score.py's query_vlm_safety() 5 times per object, at the same call
configuration already used in production (temperature=0.2).

Images are rough eyeballed bounding-box crops (assets/vlm_consistency_real/images/)
taken directly from real Replica room0 RGB frames (huggingface.co/datasets/
3David14/GS3LAM-Replica), pulled via a small in-memory HTTP-range zip reader (not the
full 12.77GB archive) -- see PROGRESS.md for the crop provenance (source frame + box
per object). This replaces the original stock-photo proxies with the pipeline's actual
target data distribution.

Only two things differ from vlm_consistency_check.py: the image source (real crops
instead of stock photos) and the model (gemini-3.5-flash, matching vlm_safety_score.py's
current production pin, instead of the experimental gemini-flash-latest rolling alias
the original script used). Prompt, temperature, query count, and generation config are
unchanged, so this is a clean comparison against the original run.
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

IMAGES_DIR = "assets/vlm_consistency_real/images"
RESULTS_PATH = "assets/vlm_consistency_real/results.json"
QUERIES_PER_OBJECT = 5

# Verbatim from vlm_safety_score.py:query_vlm_safety() -- do not edit here.
PROMPT = """
    You are a physical safety auditor for a 3kg wheeled robot (TurtleBot4).
    Analyze the physical materials, structure, and stability of the isolated object in this image.
    Output a strictly formatted JSON dictionary with a single key 'safety_score', holding a float from 0.0 (lethal hazard/fragile/easily tipped/cables) to 1.0 (completely safe to drive on/flat solid ground).
    """
TEMPERATURE = 0.2  # matches vlm_safety_score.py's existing config; already > 0.

# Pinned to the CURRENT production model (vlm_safety_score.py:34), not the
# vlm_consistency_check.py original run's 'gemini-flash-latest' rolling alias -- that
# alias predates the production pinning decision (PROGRESS.md, "Third Stage 2 bug
# found"). Using the same dated/stable release the paper will actually report.
MODEL = "gemini-3.5-flash"
MAX_RETRIES = 2

# Same infra fixes vlm_safety_score.py already carries: current "thinking" models
# silently truncate the JSON answer without these (see vlm_consistency_check.py /
# PROGRESS.md) -- not a prompt or scale change.
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


def run(image_paths, queries_per_object=QUERIES_PER_OBJECT):
    all_results = {}

    for path in image_paths:
        obj_name = os.path.splitext(os.path.basename(path))[0]
        img = PIL.Image.open(path)
        print(f"\nQuerying {obj_name} x{queries_per_object}...")

        scores = []
        for i in range(queries_per_object):
            for attempt in range(MAX_RETRIES + 1):
                try:
                    score = query_once(img)
                    scores.append(score)
                    print(f"  [{i+1}/{queries_per_object}] safety_score = {score:.3f}")
                    break
                except Exception as e:
                    if attempt < MAX_RETRIES:
                        print(f"  [{i+1}/{queries_per_object}] retry {attempt+1} after error: {e}")
                    else:
                        print(f"  [{i+1}/{queries_per_object}] ERROR (out of retries): {e}")

        all_results[obj_name] = {
            "scores": scores,
            "mean": statistics.mean(scores) if scores else None,
            "stdev": statistics.stdev(scores) if len(scores) > 1 else None,
        }

    return all_results


def print_table(all_results):
    print("\n" + "=" * 88)
    print(f"{'Object':32s} {'Scores':32s} {'Mean':>8s} {'StdDev':>8s}")
    print("-" * 88)
    for obj_name, r in all_results.items():
        scores_str = ", ".join(f"{s:.2f}" for s in r["scores"])
        mean_str = f"{r['mean']:.3f}" if r["mean"] is not None else "N/A"
        std_str = f"{r['stdev']:.3f}" if r["stdev"] is not None else "N/A"
        print(f"{obj_name:32s} {scores_str:32s} {mean_str:>8s} {std_str:>8s}")
    print("=" * 88)


def main():
    image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
    if not image_paths:
        raise FileNotFoundError(f"No images found in {IMAGES_DIR}")

    all_results = run(image_paths)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    print_table(all_results)
    print(f"\nRaw results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
