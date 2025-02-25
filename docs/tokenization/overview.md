# Tokenization in LBSTER

Tokenization is a critical preprocessing step for language models, converting raw biological sequences into numeric tokens that models can process. LBSTER provides specialized tokenizers for different types of biological sequences.

## Tokenizer Types

LBSTER includes several tokenizer implementations for different biological sequence types:

### 1. Amino Acid Tokenizer

The `AminoAcidTokenizerFast` tokenizer is designed for protein sequences and tokenizes each amino acid as a separate token.

```python
from lobster.tokenization import AminoAcidTokenizerFast

tokenizer = AminoAcidTokenizerFast()
tokens = tokenizer("MVLSPADKTNVKAAWG", return_tensors="pt")
print(tokens["input_ids"])
```

### 2. Nucleotide Tokenizer

The `NucleotideTokenizerFast` tokenizer is specialized for DNA and RNA sequences, tokenizing each nucleotide (A, C, G, T/U) as a separate token.

```python
from lobster.tokenization import NucleotideTokenizerFast

tokenizer = NucleotideTokenizerFast()
tokens = tokenizer("ATGCGATCGATCGATCG", return_tensors="pt")
print(tokens["input_ids"])
```

### 3. SMILES Tokenizer

The `SmilesTokenizerFast` tokenizer handles SMILES strings for representing chemical molecules, breaking them down into chemically meaningful tokens.

```python
from lobster.tokenization import SmilesTokenizerFast

tokenizer = SmilesTokenizerFast()
tokens = tokenizer("CC(=O)OC1=CC=CC=C1C(=O)O", return_tensors="pt")  # Aspirin
print(tokens["input_ids"])
```

### 4. PMLM Tokenizer

The `PmlmTokenizer` is the default tokenizer for LBSTER's protein language models, compatible with ESM-style tokenization.

```python
from lobster.tokenization import PmlmTokenizer
import importlib.resources

path = importlib.resources.files("lobster") / "assets" / "pmlm_tokenizer"
tokenizer = PmlmTokenizer.from_pretrained(path, do_lower_case=False)
tokens = tokenizer("MVLSPADKTNVKAAWG", return_tensors="pt")
print(tokens["input_ids"])
```

### 5. MGM Tokenizer

The `MgmTokenizer` is a Multi-modal Genomics Model tokenizer supporting multiple modalities including amino acids, nucleotides, and SELFIES.

```python
from lobster.tokenization import MgmTokenizer

tokenizer = MgmTokenizer(model_max_length=512)
tokens = tokenizer("ATGCGATCGATCGATCG", return_tensors="pt")
print(tokens["input_ids"])
```

## Tokenizer Transforms

LBSTER provides transform wrappers around tokenizers for easier integration with data pipelines:

### Basic Tokenizer Transform

```python
from lobster.transforms import TokenizerTransform
from lobster.tokenization import AminoAcidTokenizerFast

transform = TokenizerTransform(
    tokenizer=AminoAcidTokenizerFast(),
    padding="max_length",
    max_length=512,
    truncation=True
)

# Can be used with datasets
tokens = transform("MVLSPADKTNVKAAWG")
```

### Concept-aware Tokenizer Transform

```python
from lobster.tokenization import PmlmConceptTokenizerTransform
import importlib.resources

path = importlib.resources.files("lobster") / "assets" / "pmlm_tokenizer"
transform = PmlmConceptTokenizerTransform(
    path,
    padding="max_length",
    truncation=True,
    max_length=512,
    normalize=True
)

# Returns both tokens and concept values
result = transform("MVLSPADKTNVKAAWG")
tokens = result["input_ids"]
concepts = result["all_concepts"]
```

## Special Tokens

LBSTER tokenizers use several special tokens:

- `<cls>`: Start of sequence token
- `<pad>`: Padding token
- `<eos>`: End of sequence token
- `<unk>`: Unknown token
- `<mask>`: Mask token (for masked language modeling)
- `<sep>`: Separator token (for multi-sequence tasks)

Additional special tokens may be present in specific tokenizers.

## Vocabulary Sizes

Different tokenizers have different vocabulary sizes:

- Amino Acid Tokenizer: 33 tokens (20 standard amino acids + special tokens)
- Nucleotide Tokenizer: 12 tokens (4 nucleotides + special tokens)
- PMLM Tokenizer: Depends on the pre-trained model, typically 30-33 tokens
- MGM Tokenizer: Variable based on configuration

## Using Tokenizers with Models

LBSTER models are instantiated with their corresponding tokenizer:

```python
from lobster.model import LobsterPMLM

# Model comes with appropriate tokenizer
model = LobsterPMLM("asalam91/lobster_24M")
tokenizer = model.tokenizer

# Tokenize a sequence
tokens = tokenizer("MVLSPADKTNVKAAWG", return_tensors="pt")

# Forward pass through the model
outputs = model.model(input_ids=tokens["input_ids"], 
                      attention_mask=tokens["attention_mask"])
```

## Custom Tokenization Workflows

For advanced use cases, you can build custom tokenization pipelines:

```python
from lobster.tokenization import PmlmTokenizer
import importlib.resources
import torch

path = importlib.resources.files("lobster") / "assets" / "pmlm_tokenizer"
tokenizer = PmlmTokenizer.from_pretrained(path, do_lower_case=False)

def batch_tokenize(sequences, max_length=512):
    """Tokenize a batch of sequences with padding."""
    encodings = [tokenizer.encode(seq) for seq in sequences]
    
    # Determine max length in this batch
    batch_max_len = min(max(len(enc) for enc in encodings), max_length)
    
    # Pad sequences
    padded_encodings = []
    attention_masks = []
    
    for enc in encodings:
        # Truncate if necessary
        if len(enc) > max_length:
            enc = enc[:max_length]
        
        # Create attention mask (1 for tokens, 0 for padding)
        attention_mask = [1] * len(enc) + [0] * (batch_max_len - len(enc))
        attention_mask = attention_mask[:max_length]
        
        # Pad sequence
        padded_enc = enc + [tokenizer.pad_token_id] * (batch_max_len - len(enc))
        padded_enc = padded_enc[:max_length]
        
        padded_encodings.append(padded_enc)
        attention_masks.append(attention_mask)
    
    return {
        "input_ids": torch.tensor(padded_encodings),
        "attention_mask": torch.tensor(attention_masks)
    }
```

In the following sections, we'll explore each tokenizer type in more detail.
