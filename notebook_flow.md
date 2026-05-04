# Notebook flow (focus on embeddings)

This document summarizes the end-to-end flow of the notebook with extra detail on the embedding pipeline and model-specific handling.

## High-level flow

```mermaid
flowchart TD
    A[Load config + data] --> B[Clean text + fix encoding]
    B --> C[Age masking rules]
    C --> D[Split by task: greeting / trip]
    D --> E[Build full conversation per subject]
    E --> F[Preview random conversations]
    F --> G[Token length diagnostics]
    G --> H[Build masked + unmasked datasets]
    H --> I[Shared subject-level split]
    I --> J[Expand encoder variants]
    J --> K[Embedding extraction per variant]
    K --> L[Logistic Regression + CV + evaluation]
    L --> M[Comparison + plots + confusion matrix]
```

## Embedding pipeline (detailed)

```mermaid
flowchart TD
    A[Chat-level dataframe] --> B[Select version: masked / unmasked]
    B --> C[Expand encoder variants]
    C --> D[Load encoder + tokenizer]
    D --> E[Tokenize texts]
    E --> F{Longformer + GA on?}
    F -->|Yes| G[Add global_attention_mask on CLS]
    F -->|No| H[Use default attention_mask]
    G --> I[Forward pass]
    H --> I
    I --> J{Pooling}
    J -->|CLS| K[Take CLS vector]
    J -->|Mean| L[Masked mean pooling]
    J -->|Max| M[Masked max pooling]
    K --> N[Embeddings matrix]
    L --> N
    M --> N
    N --> O[StandardScaler]
    O --> P[Logistic Regression + CV + tuning]
    P --> Q[Metrics + reports]
```

## Key embedding logic

- `ENCODER_SPECS` defines each model name, `max_length`, and default global attention setting.
- `expand_embedding_variants()` generates variants for pooling (`cls`, `mean`, `max`) and Longformer GA on/off, producing `run_label` strings.
- `get_embeddings()` runs batched tokenization and forward passes, then applies pooling.
- `global_attention_mask` is only added for Longformer when `use_global_attention=True` and is applied to the CLS token.
- Embeddings are standardized and fed into Logistic Regression with cross-validation and light tuning.

## Where each stage lives in the notebook

- Data cleaning + masking: early cells after file upload.
- Conversation building: `build_full_conversation()` and `build_conversation_df()` cells.
- Token length diagnostics: token counting cell using Longformer/BigBird/LED/Long-T5/RoBERTa tokenizers.
- Balanced split: subject-level stratified split cells.
- Embedding extraction: `get_embeddings()` and `run_single_experiment()` cells.
- Evaluation: results tables, plots, heatmap, confusion matrix.

## Notes specific to Longformer

- Longformer is evaluated with global attention both on and off the CLS token.
- Pooling is tested with `cls`, `mean`, and `max`, and each variant is tracked via `run_label`.
