import os
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

# Pricing (Input / Output per 1M tokens)
PRICING = {
    "gpt-5.1": (1.25, 0.10),
    "gpt-5-mini": (0.25, 2.00),
    "gpt-4.1-mini": (0.80, 3.20),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-3-7-sonnet": (3.00, 15.00),
    "claude-4-0-sonnet": (3.00, 15.00),
    "claude-4-5-sonnet": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-3-5-haiku": (0.80, 4.00),
}

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

        # Cost Calc (Approximate)
        # Image tokens vary, but assuming ~1000 input tokens for image+text and ~500 output
        p_in, p_out = PRICING.get(model_conf["name"], (0, 0))
        est_cost = (1000 / 1_000_000 * p_in) + (500 / 1_000_000 * p_out)

        print(f"   ⏱️  Latency: {duration:.2f}s")
        print(f"   💰 Est. Cost (1 img): ${est_cost:.5f}")
        print(f"   📝 Output Length: {len(content)} chars")

        print("\n   👇 GENERATED OUTPUT 👇")
        print("-" * 60)
        print(content)
        print("-" * 60)

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")


def main():
    if not Path(TEST_IMAGE_PATH).exists():
        print(f"❌ Test image not found at {TEST_IMAGE_PATH}")
        return

    print(f"🔍 Benchmarking Vision Models on: {Path(TEST_IMAGE_PATH).name}")
    b64_img = encode_image(TEST_IMAGE_PATH)

    for model in MODELS_TO_TEST:
        test_model(model, b64_img)


if __name__ == "__main__":
    main()
