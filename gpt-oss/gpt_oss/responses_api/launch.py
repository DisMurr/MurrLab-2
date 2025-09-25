#!/usr/bin/env python3
"""
Launch the Responses API ensuring package imports work (avoids relative import errors).
Equivalent to running `python -m gpt_oss.responses_api.serve`.
"""
from .serve import *  # noqa: F401,F403
