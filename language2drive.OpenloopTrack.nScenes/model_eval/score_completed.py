#!/usr/bin/env python3
"""Compatibility entry point for the complete paper-aligned evaluator."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_predictions import main

if __name__=='__main__':main()
