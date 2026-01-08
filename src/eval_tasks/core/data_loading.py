"""
Data loading utilities for CORE evaluation tasks.

Handles loading evaluation data from JSONL files and YAML configuration files.
"""

import json
import os
from typing import List, Dict, Any

import yaml


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load a JSONL (JSON Lines) file into a list of dictionaries.

    Args:
        filepath: Path to .jsonl file

    Returns:
        List of dictionaries, one per line in the file
    """
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_core_tasks(yaml_path: str, data_root: str) -> List[Dict[str, Any]]:
    """
    Load CORE evaluation tasks from YAML configuration and corresponding data files.

    Reads a YAML config file (e.g., core.yaml) that defines evaluation tasks with fields like:
        label: arc_easy                                  # Task identifier
        dataset_uri: world_knowledge/arc_easy.jsonl      # Path to data file
        num_fewshot: [10]                                # Few-shot example counts to test
        icl_task_type: multiple_choice                   # Task type
        continuation_delimiter: "\\nAnswer: "            # Prompt delimiter

    Task Types:
    - multiple_choice: Select best completion from choices (e.g., HellaSwag, ARC)
    - language_modeling: Exact token prediction (e.g., LAMBADA, SQuAD)
    - schema: Match context options to continuation (e.g., Winograd)

    Args:
        yaml_path: Path to YAML config file defining tasks
        data_root: Root directory containing eval_data/ subdirectory

    Returns:
        List of task dictionaries, each containing:
            - label: Task name/identifier
            - task_type: One of "multiple_choice", "language_modeling", "schema"
            - num_fewshot: Number of few-shot examples to use
            - continuation_delimiter: Delimiter string for prompts
            - data: List of evaluation examples loaded from JSONL file
    """
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    tasks = []
    for task_config in config["icl_tasks"]:
        label = task_config["label"]
        dataset_uri = task_config["dataset_uri"]
        task_type = task_config["icl_task_type"]
        num_fewshot_list = task_config.get("num_fewshot", [0])
        continuation_delimiter = task_config.get("continuation_delimiter", " ")

        # Load the evaluation data from JSONL file
        data_path = os.path.join(data_root, "eval_data", dataset_uri)
        if not os.path.exists(data_path):
            print(f"Warning: Data file not found: {data_path}")
            continue

        data = load_jsonl(data_path)

        # Create a separate task entry for each few-shot configuration
        # (allows evaluating the same dataset with 0-shot, 5-shot, 10-shot, etc.)
        for num_fewshot in num_fewshot_list:
            task = {
                "label": label,
                "task_type": task_type,
                "num_fewshot": num_fewshot,
                "continuation_delimiter": continuation_delimiter,
                "data": data,
            }
            tasks.append(task)

    return tasks
