# AuGR for Sequential Recommendation

This folder contains code used to produce experimental results for AuGR on generative recommendation task with Amazon 2014 dataset. This work was built off the repository: [https://phonism.github.io/genrec](https://phonism.github.io/genrec).

All train settings for AuGR variants are the same as baseline models in the repository.

## Installation

### From Source (Recommended)

Enter this folder directory, then follow the steps below:

```bash
cd generative-recommendation/genrec
pip install -e .
```

### Full Installation (with Triton, TorchRec, etc.)

```bash
pip install -e ".[full]"
```

### Dependencies Only

```bash
pip install -r requirements.txt
```

## Quick Start

### Train Baseline Models

```bash
# SASRec on Amazon 2014
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon.gin --split beauty

# HSTU on Amazon 2014
python genrec/trainers/hstu_trainer.py config/hstu/amazon.gin --split beauty
```

### Train AuGR variants
```bash
# SASRec on Amazon 2014
python genrec/trainers/sasrec_trainer.py config/sasrec/amazon_augr.gin --split beauty

# HSTU on Amazon 2014
python genrec/trainers/hstu_trainer.py config/hstu/amazon_augr.gin --split beauty
```

### Train RQVAE (Semantic ID Generator - prerequisite for TIGER)
```bash
# For TIGER pipeline
python genrec/trainers/rqvae_trainer.py config/tiger/amazon/rqvae.gin --split beauty 
```

### Train TIGER models
```bash
# Base TIGER: Requires pretrained RQVAE checkpoint
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger.gin --split beauty

# AuGR-TIGER: Requires pretrained RQVAE checkpoint
python genrec/trainers/tiger_trainer.py config/tiger/amazon/tiger_augr_catalog_ce.gin --split beauty
```

## Documentation

Full documentation is available in the original repository: [https://phonism.github.io/genrec](https://phonism.github.io/genrec)