#!/usr/bin/env python3
"""Answer one chat message with 2DSD and report its decode throughput.

  python scripts/generate.py --model ouro-1.4b-thinking \\
      --prompt "What is the sum of the first 100 positive integers?"

--model is a registry name of configs/models.json (ouro-1.4b-thinking,
ouro-2.6b-thinking, nanbeige4.2-3b, dory1.0-4b), a local checkpoint or a
Hugging Face id. A registry name runs one of its targets with that target's
drafter: --target w64 (default) is the shared-KV target and --target original
the base model. Dory runs its base checkpoint for both, with KV policy w64
(last-loop sharing) or native (per-loop KV), and one drafter. The registry
pins the base models' revisions. Explicit flags override the registry:
--revision sets the revision of the selected target's repository, --drafter
replaces the registry's drafter, and --kv-policy (Dory only) the policy that
--target sets. With --block-size 1 no drafter runs. Any other --model runs as
given, without --target: pass --drafter, --revision as needed, and --kv-policy
for a Dory checkpoint (default w64).

The prompt is rendered as scripts/build_prompts.py renders the benchmark
prompts: with the chat template's defaults (no <think> for Ouro, whose template
adds it only on request; the Nanbeige and Dory templates open the thinking
block) and, unless --system is given, the family's system turn (the paper's
English one for Nanbeige, the template default otherwise). Use --temperature 0.6
--top-p 0.95 to sample.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # training.common.registry reads configs/models.json
from training.common import registry  # noqa: E402
from training.common.chat import system_messages  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    """Command-line arguments with --model resolved to the target, revision, drafter and KV policy to run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help=f"registry name ({', '.join(registry.names())}), local checkpoint or Hugging Face id")
    parser.add_argument("--target", choices=("w64", "original"),
                        help="registry names only: w64 shared-KV target (default) or original base model")
    parser.add_argument("--drafter", help="drafter export (local path, Hugging Face id, or id@revision); "
                        "overrides a registry name's drafter")
    parser.add_argument("--revision", help="revision of the target repository (replaces a base model's pin)")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--system", help="system message (default: the benchmark prompts' system turn)")
    parser.add_argument("--max-total-tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--block-size", type=int, help="override the block size (1: no drafter)")
    parser.add_argument("--kv-policy", choices=("w64", "native"),
                        help="Dory only: w64 last-loop sharing or native per-loop KV; set by --target for "
                        "dory1.0-4b, w64 by default for a Dory checkpoint given by path or id")
    args = parser.parse_args(argv)
    try:
        resolved = registry.serving(args.model, args.target, revision=args.revision, drafter=args.drafter,
                                    kv_policy=args.kv_policy, with_drafter=args.block_size != 1)
    except ValueError as error:
        parser.error(str(error))
    resolved.pop("family")
    vars(args).update(resolved)
    if args.drafter is None and args.block_size != 1:
        parser.error("2DSD needs --drafter (or --block-size 1)")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    print(f"target {args.model}" + (f"@{args.revision}" if args.revision else "")
          + f", drafter {args.drafter}, kv_policy {args.kv_policy or 'from config.json'}", file=sys.stderr)
    from vllm.v1.spec_decode.twodsd import TwoDSD

    engine = TwoDSD(args.model, args.drafter, max_model_len=args.max_total_tokens,
                    block_size=args.block_size, revision=args.revision, kv_policy=args.kv_policy)
    system = [{"role": "system", "content": args.system}] if args.system else system_messages(engine.model_type)
    messages = [*system, {"role": "user", "content": args.prompt}]
    text = engine.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    prompt_ids = engine.tokenizer.encode(text, add_special_tokens=False)
    [result] = engine.generate([prompt_ids], max_total_tokens=args.max_total_tokens,
                               temperature=args.temperature, top_p=args.top_p, warmup_requests=1)
    output = result["output_ids"]
    print(engine.tokenizer.decode(output, skip_special_tokens=False))
    stats = result["stats"]
    print(
        f"\n[{len(output)} tokens, decode {(len(output) - 1) / result['decode_seconds']:.1f} tok/s, "
        f"{stats['calls']} target calls, {(len(output) - 1) / max(stats['calls'], 1):.2f} tokens/call]"
    )


if __name__ == "__main__":
    main()
