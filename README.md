# Condor

A Code Discriminator Integrating General Semantics with Code Details

Paper: https://arxiv.org/abs/2412.17429

<img width="970" alt="image" src="https://github.com/user-attachments/assets/40aa1c37-f187-4292-bc97-5ea18cc68800">

The Condor consists of two main components: contrastive learning at the embedding level to capture code details (upper section), and data-level augmentation through intermediate code, which supplements code details that are not recorded in existing datasets (lower section).


# Dataset
<img width="951" alt="image" src="https://github.com/user-attachments/assets/4a74a5d2-7046-41a4-a300-6b9354ab3478">

You can find the CodeNanoFix dataset from `/data`.

# Models
You can find the model with fine-tuned weights from `/models`.

# Fine-tuning
You can find the training details from `/src`.

Training entry -> GoGoGoxx.py

```
python GoGoGoSingleDS.py
```
