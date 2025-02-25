# LBSTER Models Overview

LBSTER provides several types of pre-trained language models for biological sequences, each designed for different purposes. This page provides an overview of the available model architectures and pre-trained weights.

## Model Architectures

LBSTER implements several core model architectures:

### 1. Masked Language Models (LobsterPMLM)

These models are based on the BERT architecture with masked language modeling objectives. They are designed for bidirectional representation learning of protein sequences. 

**Use cases**:
- Protein sequence embedding
- Protein property prediction
- Sequence classification
- Sequence similarity search

**Implementation**: `LobsterPMLM` class

```python
from lobster.model import LobsterPMLM
model = LobsterPMLM("asalam91/lobster_24M")
```

### 2. Concept Bottleneck Models (LobsterCBMPMLM)

These models extend masked language models with an interpretable concept bottleneck layer. They can encode proteins into interpretable biological concepts while maintaining performance on downstream tasks.

**Use cases**:
- Interpretable protein representation learning
- Controllable protein generation
- Concept-guided sequence editing

**Implementation**: `LobsterCBMPMLM` class

```python
from lobster.model import LobsterCBMPMLM
model = LobsterCBMPMLM("asalam91/cb_lobster_24M")
```

### 3. Causal Language Models (LobsterPCLM)

These models are based on the auto-regressive decoder-only architecture similar to GPT models. They are designed for generating biological sequences token by token.

**Use cases**:
- Protein sequence generation
- Protein design
- Sequence completion

**Implementation**: `LobsterPCLM` class

```python
from lobster.model import LobsterPCLM
model = LobsterPCLM.load_from_checkpoint("path/to/checkpoint.ckpt")
```

### 4. Structure Prediction Models (LobsterPLMFold)

These models combine language model pre-training with structure prediction heads, allowing for predicting protein structure from sequence.

**Use cases**:
- Structure prediction
- Structure-guided sequence design

**Implementation**: `LobsterPLMFold` class

## Pre-trained Models

LBSTER provides several pre-trained models of different sizes and training objectives:

### Masked Language Models

| Model | Parameters | Dataset | Description | Link |
|-------|------------|---------|-------------|------|
| lobster_24M | 24M | UniRef50 | 24M parameter protein Masked LLM | [lobster_24M](https://huggingface.co/asalam91/lobster_24M) |
| lobster_150M | 150M | UniRef50 | 150M parameter protein Masked LLM | [lobster_150M](https://huggingface.co/asalam91/lobster_150M) |

### Concept Bottleneck Models

| Model | Parameters | Dataset | # Concepts | Link |
|-------|------------|---------|------------|------|
| cb_lobster_24M | 24M | UniRef50+SwissProt | 718 | [cb_lobster_24M](https://huggingface.co/asalam91/cb_lobster_24M) |
| cb_lobster_150M | 150M | UniRef50+SwissProt | 718 | [cb_lobster_150M](https://huggingface.co/asalam91/cb_lobster_150M) |
| cb_lobster_650M | 650M | UniRef50+SwissProt | 718 | [cb_lobster_650M](https://huggingface.co/asalam91/cb_lobster_650M) |
| cb_lobster_3B | 3B | UniRef50+SwissProt | 718 | [cb_lobster_3B](https://huggingface.co/asalam91/cb_lobster_3B) |

## Model Selection Guide

When choosing which LBSTER model to use, consider:

1. **Task requirements**: 
   - For embeddings or protein property prediction → Masked models
   - For controllable generation → Concept Bottleneck models
   - For pure generation → Causal models
   - For structure prediction → PLMFold models

2. **Compute constraints**:
   - Limited GPU memory (≤8GB) → 24M parameter models
   - Medium GPU memory (≤16GB) → 150M parameter models
   - High GPU memory (≥24GB) → 650M or 3B parameter models

3. **Speed vs. accuracy tradeoff**:
   - Fastest inference → 24M parameter models
   - Best accuracy → 3B parameter models

## Usage Examples

### Loading a pre-trained model:

```python
# Masked language model
from lobster.model import LobsterPMLM
mlm = LobsterPMLM("asalam91/lobster_24M")

# Concept bottleneck model
from lobster.model import LobsterCBMPMLM
cbm = LobsterCBMPMLM("asalam91/cb_lobster_24M")

# Causal language model
from lobster.model import LobsterPCLM
clm = LobsterPCLM.load_from_checkpoint("path/to/checkpoint.ckpt")
```

### Basic inference:

```python
# Encoding a sequence with a masked language model
sequence = "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF"
tokens = mlm.tokenizer(sequence, return_tensors="pt")
embeddings = mlm.model(input_ids=tokens["input_ids"], 
                        attention_mask=tokens["attention_mask"])
cls_embedding = embeddings[:, 0, :]  # Get the [CLS] token embedding

# Getting concepts from a concept bottleneck model
tokens = cbm.tokenizer(sequence, return_tensors="pt")
outputs = cbm.model(input_ids=tokens["input_ids"], 
                    attention_mask=tokens["attention_mask"],
                    inference=True)
concepts = outputs["concepts"]  # Get the concept values
```

In the following sections, we'll explore each model type in more detail.
