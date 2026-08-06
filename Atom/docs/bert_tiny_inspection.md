# BERT Tiny adapter-target inspection

Captured with:

```powershell
python scripts/inspect_model.py
```

Model: `prajjwal1/bert-tiny`

| Property | Measured value |
|---|---:|
| Transformer layers | 2 |
| Hidden size | 128 |
| Linear modules | 13 |
| Query targets | 2 |
| Value targets | 2 |
| Total adapter targets | 4 |
| Base-model parameters | 4,385,920 |

The four matched `nn.Linear` modules are:

| Module | Weight shape |
|---|---:|
| `encoder.layer.0.attention.self.query` | `128 x 128` |
| `encoder.layer.0.attention.self.value` | `128 x 128` |
| `encoder.layer.1.attention.self.query` | `128 x 128` |
| `encoder.layer.1.attention.self.value` | `128 x 128` |

The injector must match the complete suffixes `attention.self.query` and
`attention.self.value`. Keys, output projections, feed-forward layers, the pooler, and all other
linear modules remain unadapted.
