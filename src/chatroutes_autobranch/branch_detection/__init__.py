"""
Branch detection module for identifying decision points in text.

This module provides tools to analyze text (like LLM responses) and identify
branch points where multiple mutually-exclusive options exist.
"""

from chatroutes_autobranch.branch_detection.models import (
    BranchOption,
    BranchPoint,
)
from chatroutes_autobranch.branch_detection.extractor import BranchExtractor
from chatroutes_autobranch.branch_detection.parser import LLMBranchParser

__all__ = [
    "BranchOption",
    "BranchPoint",
    "BranchExtractor",
    "LLMBranchParser",
]
