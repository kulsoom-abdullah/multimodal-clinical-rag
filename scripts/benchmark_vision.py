import os
import json
import time
import base64
from pathlib import Path
from dotenv import load_dotenv

# Model Interfaces
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

load_dotenv()

# --- CONFIGURATION ---
TEST_IMAGE_PATH = "images/benchmark_heatmap.jpeg"

# Published list prices, USD per 1M tokens (input, output). This is config, not
# measurement -- costs below are these rates applied to the tokens a call actually
# used. Re-check against the providers' pricing pages before quoting any figure;
# there is no way for this script to detect a stale rate.
#
# validate_pricing() flags entries where output costs less than input, which no
# current provider charges and which silently understates a model's cost.
#
# OpenAI rates verified against openai.com pricing on 2026-08-04.
# Anthropic rates not re-verified on that date.
PRICING = {
    # 2026-08-04: corrected from (1.25, 0.10). The old output rate was the
    # cached-input price with a dropped decimal, understating output 100x.
    "gpt-5.1": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4.1-mini": (0.80, 3.20),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-4-0-sonnet": (3.00, 15.00),
    "claude-4-5-sonnet": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # Key must match the "name" in MODELS_TO_TEST -- it was "claude-3-5-haiku"
    # while the model is listed as "claude-haiku-3-5", so the lookup missed and
    # the old code's (0, 0) default reported this model as costing nothing.
    "claude-haiku-3-5": (0.80, 4.00),
}

# Collected per model, then written as a run artifact.
results = []

MODELS_TO_TEST = [
    # OpenAI
    {"name": "gpt-5.1", "provider": "openai", "model_id": "gpt-5.1"},
    {"name": "gpt-5-mini", "provider": "openai", "model_id": "gpt-5-mini"},
    {"name": "gpt-4.1-mini", "provider": "openai", "model_id": "gpt-4.1-mini"},
    {"name": "gpt-4o-mini", "provider": "openai", "model_id": "gpt-4o-mini"},
    # Anthropic (Specific Date Versions are safer than 'latest')
    {
        "name": "claude-haiku-4-5",
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
    },
    {
        "name": "claude-haiku-3-5",
        "provider": "anthropic",
        "model_id": "claude-3-5-haiku-20241022",
    },
    {
        "name": "claude-3-7-sonnet",
        "provider": "anthropic",
        "model_id": "claude-3-7-sonnet-20250219",
    },
    {
        "name": "claude-4-0-sonnet",
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-20250514",
    },
    {
        "name": "claude-4-5-sonnet",
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-5-20250929",
    },
]


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def test_model(model_conf, base64_image):
    print(f"\n🥊 TESTING: {model_conf['name']}...")

    try:
        # Init Model
        if model_conf["provider"] == "openai":
            llm = ChatOpenAI(
                model=model_conf["model_id"], max_tokens=1000, temperature=0
            )
        elif model_conf["provider"] == "anthropic":
            llm = ChatAnthropic(
                model=model_conf["model_id"], max_tokens=1000, temperature=0
            )

        prompt_text = """
        CONTEXT: Clinical Trial ID NCT02423343. 
        INSTRUCTIONS: Describe this figure in detail for a search engine index. 
        If redacted, mention it but describe visible data. 
        """

        msg = HumanMessage(
            content=[
                {"type": "text", "text": prompt_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                },
            ]
        )

        # Timing
        start_time = time.time()
        response = llm.invoke([msg])
        end_time = time.time()

        duration = end_time - start_time
        content = response.content.strip()

        # Cost from the tokens this call actually used, reported by the provider.
        # Previously both counts were hardcoded (1000 in / 500 out) for every model,
        # which made the cost column a restatement of the price list: it would have
        # produced identical numbers without the benchmark ever running, and it
        # billed models that returned nothing as though they had emitted 500 tokens.
        usage = getattr(response, "usage_metadata", None) or {}
        tok_in = usage.get("input_tokens")
        tok_out = usage.get("output_tokens")

        p_in, p_out = PRICING.get(model_conf["name"], (None, None))
        if p_in is None:
            cost, cost_note = None, "no price on file"
        elif tok_in is None or tok_out is None:
            cost, cost_note = None, "provider returned no token usage"
        elif not content:
            # Empty reply: report the input actually paid for, not a notional output.
            cost, cost_note = tok_in / 1e6 * p_in, "empty response, input only"
        else:
            cost, cost_note = (tok_in / 1e6 * p_in) + (tok_out / 1e6 * p_out), ""

        print(f"   ⏱️  Latency: {duration:.2f}s")
        print(f"   🔢 Tokens: in={tok_in} out={tok_out}")
        print(
            f"   💰 Cost (1 img): "
            + (f"${cost:.6f}" if cost is not None else "n/a")
            + (f"  ({cost_note})" if cost_note else "")
        )
        print(f"   📝 Output Length: {len(content)} chars")

        results.append(
            {
                "model": model_conf["name"],
                "latency_s": round(duration, 2),
                "input_tokens": tok_in,
                "output_tokens": tok_out,
                "cost_usd": cost,
                "output_chars": len(content),
                "status": "empty_response" if not content else "ok",
                "note": cost_note,
            }
        )

        print("\n   👇 GENERATED OUTPUT 👇")
        print("-" * 60)
        print(content)
        print("-" * 60)

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        results.append(
            {
                "model": model_conf["name"],
                "latency_s": None,
                "input_tokens": None,
                "output_tokens": None,
                "cost_usd": None,
                "output_chars": 0,
                "status": "error",
                "note": str(e)[:200],
            }
        )


def validate_pricing():
    """Flag price entries that cannot be right, and models with no price at all."""
    for m in MODELS_TO_TEST:
        name = m["name"]
        if name not in PRICING:
            print(f"⚠️  no price on file for {name}; its cost will report as n/a")
            continue
        p_in, p_out = PRICING[name]
        if p_out < p_in:
            print(
                f"⚠️  {name}: output rate (${p_out}/1M) is below input (${p_in}/1M). "
                "No current provider prices this way -- verify before quoting its cost."
            )


def main():
    if not Path(TEST_IMAGE_PATH).exists():
        print(f"❌ Test image not found at {TEST_IMAGE_PATH}")
        return

    validate_pricing()
    print(f"🔍 Benchmarking Vision Models on: {Path(TEST_IMAGE_PATH).name}")
    b64_img = encode_image(TEST_IMAGE_PATH)

    for model in MODELS_TO_TEST:
        test_model(model, b64_img)

    # Write a run artifact. Results were previously printed and lost, so every
    # figure quoted from this benchmark was unreproducible without paying to re-run
    # it across two providers.
    out = Path("data/eval_runs") / "vision_benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "image": TEST_IMAGE_PATH,
                "note": (
                    "Costs are published list prices applied to the tokens each call "
                    "actually used. Quality is measured only as output length; there "
                    "is no accuracy rubric. One image."
                ),
                "pricing_usd_per_1m": PRICING,
                "results": results,
            },
            indent=2,
        )
    )
    print(f"\n💾 Wrote {out}")

    print(f"\n{'MODEL':<20} {'LATENCY':>8} {'IN':>7} {'OUT':>7} {'COST':>11} {'CHARS':>7}")
    for r in results:
        c = f"${r['cost_usd']:.6f}" if r["cost_usd"] is not None else "n/a"
        print(
            f"{r['model']:<20} {str(r['latency_s'] or '-'):>8} {str(r['input_tokens'] or '-'):>7} "
            f"{str(r['output_tokens'] or '-'):>7} {c:>11} {r['output_chars']:>7}"
        )


if __name__ == "__main__":
    main()
