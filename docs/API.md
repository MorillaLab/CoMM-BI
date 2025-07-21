# CoMM-BIP API Reference

## Core Classes

### `CoMM_BIP` (Model Class)
```python
class CoMM_BIP(pl.LightningModule):
    """
    Multimodal model integrating transcriptomic, metabolomic, phenomic, 
    and environmental data with biological priors.
    """
    def __init__(
        self,
        input_dims: Dict[str, int],  # e.g., {'rna': 1000, 'metabo': 50}
        feature_names: Dict[str, List[str]],
        temperature: float = 0.1
    ):
        """
        Args:
            input_dims: Dictionary of input dimensions per modality.
            feature_names: Feature names for biological prior masking.
            temperature: Initial value for calibration scaling.
        """
```

#### Key Methods:
| Method | Description |
|--------|-------------|
| `forward(rna, metabo, pheno, env)` | Returns logits for effector prediction. |
| `encode(rna, metabo, pheno, env)` | Extracts multimodal embeddings. |
| `training_step(batch, batch_idx)` | Joint contrastive + classification loss. |

---

### `PlantDataModule` (Data Handler)
```python
class PlantDataModule(pl.LightningDataModule):
    """
    Handles multimodal data loading and augmentation.
    """
    def prepare_data(self):
        """Loads TSV files or generates dummy data."""

    def setup(self, stage=None):
        """Splits data into train/val/test sets."""
```

#### Data Fields:
- `transcriptome`: Gene expression matrix (samples × genes)
- `metabolomics`: Metabolite abundance matrix  
- `phenomics`: Phenotypic measurements  
- `environment`: Genotype/effector labels  

---

## Functions
### `train_model()`
```python
def train_model() -> Dict:
    """Full training pipeline. Returns model, datamodule, and test results."""
```

### `calibrate_model(model, val_loader)`
```python
def calibrate_model(model, val_loader):
    """Performs temperature scaling to calibrate output probabilities."""
```

---

## Example Usage
```python
from src.model import CoMM_BIP, PlantDataModule

# Initialize
dm = PlantDataModule()
dm.setup()
model = CoMM_BIP(
    input_dims={'rna': 1000, 'metabo': 50, 'pheno': 3, 'env': 1},
    feature_names=dm.feature_names
)

# Train
trainer = pl.Trainer(max_epochs=100)
trainer.fit(model, dm)

# Predict
logits = model(rna_tensor, metabo_tensor, pheno_tensor, env_tensor)
```
