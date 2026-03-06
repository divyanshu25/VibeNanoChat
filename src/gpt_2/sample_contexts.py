"""
Sample contexts for model generation during evaluation.
This file contains different types of prompts to test various model capabilities.
"""

# Sample contexts for SFT and mid-training modes
SFT_SAMPLE_CONTEXTS = [
    "<|bos|><|user_start|>Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?<|user_end|><|assistant_start|>",
    "<|bos|><|user_start|>Write a short poem about artificial intelligence.<|user_end|><|assistant_start|>",
    "<|bos|><|user_start|>Explain the concept of recursion in programming with a simple example.<|user_end|><|assistant_start|>",
    "<|bos|><|user_start|>What are the three laws of thermodynamics?<|user_end|><|assistant_start|>",
    "<|bos|><|user_start|>A train travels 120 miles in 2 hours. What is its average speed?<|user_end|><|assistant_start|>",
    "<|bos|><|user_start|>Translate the following to French: 'Hello, how are you today?'<|user_end|><|assistant_start|>",
    "<|bos|><|user_start|>Write a function in Python that calculates the factorial of a number.<|user_end|><|assistant_start|>",
    "<|bos|><|user_start|>What is the capital of Australia and why was it chosen?<|user_end|><|assistant_start|>",
]

# Sample contexts for general pre-training mode
GENERAL_SAMPLE_CONTEXTS = [
    "Hello, I'm a language model,",
    "Once upon a time, in a land far away,",
    "The history of artificial intelligence began",
    "In the year 2025, scientists discovered",
    "def fibonacci(n):\n    ",
    "The most important lesson I learned was",
    "Climate change is affecting our planet in many ways:",
    "To build a neural network from scratch, you need to",
]
