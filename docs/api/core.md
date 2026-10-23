# Core types

::: attnbank.core
    options:
      members: [AttentionMethod, AttentionOutput, Category, Objective, check_qkv,
                causal_mask, query_positions, scores, masked_softmax, renormalize,
                normalize_nonnegative, weights_from_log_magnitude, make_generator]

## The reference operator

::: attnbank.reference
    options:
      members: [SoftmaxAttention, reference_weights]

## Shared components

The pieces a method reuses, so a method file shows what it *adds*.

::: attnbank.components
    options:
      members: []

::: attnbank.components.online_softmax
::: attnbank.components.quantization
::: attnbank.components.restriction
::: attnbank.components.recurrence
::: attnbank.components.sdpa
