# Model artifacts

This folder holds the files the ML service loads at startup.

| File | In this repository | Description |
|---|---|---|
| `model_info.json` | Yes | BiLSTM-Attention configuration, class list and test metrics |
| `gcn_info.json` | Yes | Multi-relational GCN configuration, learned relation weights and clustering metrics |
| `bilstm.pt`, `bilstm_best.pt` | No | BiLSTM-Attention weights |
| `tokenizer.pkl`, `label_encoder.pkl` | No | Character tokenizer and label encoder |
| `gcn.pt`, `gcn_best.pt` | No | GCN weights |

The trained weights are available on request. They can also be reproduced with
`training/train_bilstm.py` and `ml-service/train_gcn.py`.
