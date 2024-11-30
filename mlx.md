# Using MLX from Apple

[MLX](https://github.com/ml-explore/mlx) is an array framework for machine learning research on Apple silicon

We will be using MLX to fine-tune LLMs like Mistral, TinyLLama and others during the process of learning howto.

## Fine-tuning

I don't have a specific dataset to use to fine-tune an LLM so I'm going to use llama2:70b to generate instructions.


```
Please list in JSON format 100 frequently asked questions about Kubernetes from all levels of users working on Kubernetes.  The questions should start with any of the following: “Where do I", "Is it okay to", "Can you help me", "I need to", "Is there a", "Do you know", "Where is the", "Can you tell me", "Can I change", "What are the", "How do I", "When is it", "Does Kubernetes have", "How to", "What is the difference", "Can users", "Can I", "What is”.  You do not need to provide an answer or category to each question. The list should be a single dimension array of only questions.
```



Please list in JSON format 100 frequently asked questions about beekeeping from all levels of beekeepers.  The questions should start with any of the following: “Where do I", "Is it okay to", "Can you help me", "I need to", "Is there a", "Do you know", "Where is the", "Can you tell me", "Can I change", "What are the", "How do I", "When is it", "Does beekeeping have", "How to", "What is the difference", "Can beekeepers", "Can I", "What is”.  You do not need to provide an answer or category to each question. The list should be a single dimension array of only questions.
