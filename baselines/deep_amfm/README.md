# Deep AM-FM Baseline

## Adequacy Metric

This component aims to assess the semantic aspect of system responses, more specifically, how much source information is preserved by the dialogue generation with reference to human-written responses. The continuous space model is adopted for evaluating adequacy where good word-level or sentence-level embedding techniques are studied to measure the semantic closessness of system responses and human references in the continous vector space.

## Fluency Metric

This component aims to assess the syntactic validity of system responses. It tries to compare the system hypotheses against human references in terms of their respective sentence-level normalized log probabilities based on the assumption that sentences that are of similar syntactic validity should share similar perplexity level given by a language model. Hence, in this component, various language model techniques are explored to accurately estimate the sentence-level probability distribution.

## System Requirements

1. python 3.x
2. scipy
3. pytorch=1.6.0
4. transformers=3.5.0
