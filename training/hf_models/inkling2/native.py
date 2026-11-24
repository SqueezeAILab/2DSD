# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 The 2DSD Authors
"""The Inkling 2 reference forward, which lives in the fork next to the tree runner that executes it:
vllm.model_executor.models.inkling2_reference. This name is an alias of that module."""
import sys

from vllm.model_executor.models import inkling2_reference

sys.modules[__name__] = inkling2_reference
