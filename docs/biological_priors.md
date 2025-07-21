# Biological Priors in CoMM-BIP

## 1. `BiologicalPriorLayer`
Hardcodes domain knowledge by upweighting features with known biological relevance:

### Transcriptomic Priors
| Gene Family | Weight | Biological Role |
|-------------|--------|-----------------|
| `VIT`, `FER`, `IRT` | 2.0 | Iron transport |
| `JAZ`, `MYC`, `COI` | 1.5 | Jasmonate signaling |
| `U2AF`, `SR45` | 1.8 | RNA splicing |

### Metabolomic Priors
| Metabolite | Weight | Role |
|------------|--------|------|
| JA-Ile, SA | 2.0 | Defense hormones |
| Tomatine | 1.5 | Antimicrobial |
| Citrate, Malate | 1.3 | TCA cycle |

**Implementation**:
```python
# In model.py
self.register_buffer('mask', self._create_mask(feature_names, prior_type))
```

---

## 2. `PathwayGuidedAttention`
Learns cross-modal interactions with biological constraints:

### Key Pathways Modeled
| Interaction Type | Example | Attention Score |
|------------------|---------|-----------------|
| Transcriptome → Metabolome | `VIT` → Citrate | `R=0.82` (p<0.001) |
| Phenome → Environment | Root length → Genotype | Learned adaptively |

**Equation**:
```
Attention(Q, K, V) = softmax(QK^T * σ(bio_scale) / √d) V
```
Where `bio_scale` is a learnable parameter initialized to 1.0.

---

## 3. Validation of Biological Relevance
- **SHAP analysis** confirms that prior-weighted features (e.g., `VIT`) dominate predictions.  
- **Ablation studies** show +21% F1 score when priors are enabled (Fig 3D in paper).  

> **Note**: Customize priors by editing `BiologicalPriorLayer._create_mask()`.
