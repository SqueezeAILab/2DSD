"""argparse with a JSON file of defaults (``--config``); explicit flags override the file."""
from __future__ import annotations

import json
from pathlib import Path


def parse_with_config(parser, argv=None):
    """Parse argv after loading the optional ``--config`` JSON as parser defaults.

    Keys are argparse destinations (``replay_batch_tokens``, not ``--replay-batch-tokens``); keys
    starting with ``_`` are comments. Unknown keys are an error, so a config cannot silently miss.
    """
    known, _ = parser.parse_known_args(argv)
    if getattr(known, "config", None):
        values = json.loads(Path(known.config).read_text())
        values = {k: v for k, v in values.items() if not k.startswith("_")}
        unknown = set(values) - {action.dest for action in parser._actions}
        if unknown:
            parser.error(f"{known.config}: unknown keys {sorted(unknown)}")
        parser.set_defaults(**values)
    return parser.parse_args(argv)
