"""EarthChange AI — Interactive / Command-Line Demonstration.

Runs the complete end-to-end agentic vision-language pipeline:
Question → NLP Parsing → Plan → CV Execution → Evidence → Grounded Answer

Usage:
    python demo.py
    python demo.py --question "Where did changes occur?"
    python demo.py --question "What percentage of the upper left area changed?"
    python demo.py --interactive
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from earthchange.agent import analyze


def create_sample_satellite_images() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a realistic synthetic bitemporal satellite image pair (3 bands: RGB, 120x120).

    Simulates pre-event landscape and post-event changes in the upper-left
    and center regions (e.g. new construction or clearing).
    """
    np.random.seed(42)
    height, width = 120, 120

    # Base temporal landscape (3 bands: R, G, B)
    # Background: green/vegetation baseline
    image_a = np.empty((3, height, width), dtype=np.float32)
    image_a[0] = 50.0 + np.random.normal(0, 5, (height, width))   # Red (low)
    image_a[1] = 120.0 + np.random.normal(0, 8, (height, width))  # Green (high)
    image_a[2] = 40.0 + np.random.normal(0, 5, (height, width))   # Blue (low)

    # Time 2: same background with slight natural radiometric drift
    image_b = image_a.copy() + np.random.normal(0, 3, (3, height, width))

    # Ground truth change mask
    ground_truth = np.zeros((height, width), dtype=bool)

    # Simulate change 1: clearing / building in upper-left (rows 15:45, cols 15:45)
    image_b[0, 15:45, 15:45] = 180.0  # Bright concrete / soil
    image_b[1, 15:45, 15:45] = 175.0
    image_b[2, 15:45, 15:45] = 160.0
    ground_truth[15:45, 15:45] = True

    # Simulate change 2: small infrastructure update in center (rows 50:70, cols 50:70)
    image_b[0, 50:70, 50:70] = 140.0
    image_b[1, 50:70, 50:70] = 135.0
    image_b[2, 50:70, 50:70] = 130.0
    ground_truth[50:70, 50:70] = True

    return image_a, image_b, ground_truth


def run_pipeline(question: str, output_dir: Path | None = None) -> None:
    """Run analysis pipeline and print structured results."""
    print("=" * 70)
    print(f"Question: \"{question}\"")
    print("=" * 70)

    image_a, image_b, ground_truth = create_sample_satellite_images()

    # Spatial resolution: 10m Ground Sample Distance (typical Sentinel-2 RGB)
    pixel_res = (10.0, 10.0)

    result = analyze(
        image_a=image_a,
        image_b=image_b,
        question=question,
        pixel_resolution=pixel_res,
        output_dir=output_dir,
        ground_truth=ground_truth,
    )

    print("\n[1] NLP Understanding:")
    print(f"  • Classified Intent : {result.intent}")
    print(f"  • Target Region     : {result.region}")
    print(f"  • Operation         : {result.operation}")

    print("\n[2] Agent Execution Plan:")
    for i, step in enumerate(result.plan_steps, 1):
        print(f"  {i}. {step}")

    print("\n[3] Structured Evidence Package:")
    stats = result.evidence.summary_dict()
    total_pixels = stats.get("total_pixels")
    changed_pixels = stats.get("changed_pixels")
    print(f"  • Total Pixels      : {total_pixels:,}" if total_pixels is not None else "  • Total Pixels      : N/A")
    print(f"  • Changed Pixels    : {changed_pixels:,}" if changed_pixels is not None else "  • Changed Pixels    : N/A")
    print(f"  • Changed Percentage: {stats.get('changed_percentage', 0.0):.2f}%")
    if stats.get("changed_area_m2") is not None:
        area_m2 = stats["changed_area_m2"]
        print(f"  • Changed Area      : {area_m2:,.0f} m² ({area_m2 / 10_000:.2f} ha)")

    regions = stats.get("regions", {})
    if regions:
        print("  • Regional Breakdown:")
        for r_name in ("upper_left", "upper_right", "lower_left", "lower_right", "center"):
            if r_name in regions:
                r_stat = regions[r_name]
                print(
                    f"     - {r_name:<12}: {r_stat['changed_pixels']:>5} px "
                    f"({r_stat['changed_percentage']:>5.1f}%)"
                )

    print("\n[4] Grounded Answer (Generated strictly from evidence):")
    print("-" * 70)
    print(result.answer)
    print("-" * 70)

    if output_dir:
        vis_path = output_dir / "visualization.png"
        if vis_path.exists():
            print(f"\n[5] Visualization saved to: {vis_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EarthChange AI Demonstration")
    parser.add_argument(
        "--question",
        type=str,
        default="What percentage of the upper left area changed?",
        help="Natural language question to ask",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/outputs/demo"),
        help="Directory to save visual evidence",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive prompt mode",
    )
    args = parser.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.interactive:
        print("\nWelcome to EarthChange AI Interactive Console!")
        print("Type your questions about satellite image changes. Type 'exit' to quit.\n")
        preset_questions = [
            "Is there any change between the images?",
            "What percentage of the upper left area changed?",
            "Where did changes occur?",
            "Compare the top and bottom regions",
            "What is the total changed area?",
        ]
        print("Sample questions you can try:")
        for q in preset_questions:
            print(f"  • {q}")
        print()

        while True:
            try:
                user_q = input("\nEnter question: ").strip()
                if not user_q:
                    continue
                if user_q.lower() in ("exit", "quit", "q"):
                    print("Exiting. Have a great day!")
                    break
                run_pipeline(user_q, output_dir=out_dir)
            except (KeyboardInterrupt, EOFError):
                break
    else:
        run_pipeline(args.question, output_dir=out_dir)


if __name__ == "__main__":
    main()
