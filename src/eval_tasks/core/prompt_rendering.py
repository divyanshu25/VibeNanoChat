"""
Prompt rendering utilities for CORE evaluation tasks.

Handles the creation of prompts for different task types:
- Multiple Choice: Same context, different answer options
- Schema: Different contexts, same continuation
- Language Modeling: Context with optional continuation
"""

from typing import List, Dict, Any, Optional

from jinja2 import Template


def render_prompts_mc(
    item: Dict[str, Any],
    continuation_delimiter: str,
    fewshot_examples: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Render complete prompts for a multiple choice question.

    Creates one prompt per choice option, each with the same few-shot examples
    and query context but different answer continuations.

    Args:
        item: The question item with 'query', 'choices', and 'gold' fields
        continuation_delimiter: String to separate context from answer (e.g., "\nAnswer: ")
        fewshot_examples: Optional list of example items for in-context learning

    Returns:
        List of complete prompts, one for each choice option
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    prompts = [template.render(choice=choice, **context) for choice in item["choices"]]
    return prompts


def render_prompts_schema(
    item: Dict[str, Any],
    continuation_delimiter: str,
    fewshot_examples: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Render complete prompts for a schema question.

    Creates one prompt per context option, each with the same continuation
    but different context. Used for tasks like Winograd Schema Challenge.

    Args:
        item: The question item with 'context_options', 'continuation', and 'gold' fields
        continuation_delimiter: String to separate context from continuation
        fewshot_examples: Optional list of example items for in-context learning

    Returns:
        List of complete prompts, one for each context option
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    prompts = [
        template.render(context=context_option, **context)
        for context_option in item["context_options"]
    ]
    return prompts


def render_prompts_lm(
    item: Dict[str, Any],
    continuation_delimiter: str,
    fewshot_examples: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    """
    Render complete prompts for a language modeling task.

    Returns two prompts: one without and one with the continuation. This allows
    us to identify which tokens belong to the continuation for exact match evaluation.

    Note: We trim whitespace from contexts to avoid tokenization issues where trailing
    spaces might get absorbed into the next token, breaking prefix matching.

    Args:
        item: The question item with 'context' and 'continuation' fields
        continuation_delimiter: String to separate context from continuation
        fewshot_examples: Optional list of example items for in-context learning

    Returns:
        List of two prompts: [prompt_without_continuation, prompt_with_continuation]
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        "fewshot_examples": fewshot_examples,
        "continuation_delimiter": continuation_delimiter,
        "item": item,
    }
    # Return two prompts: without and with the continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Strip to ensure clean prefix matching in token space
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]
