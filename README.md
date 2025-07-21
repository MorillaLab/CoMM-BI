# Article
Multimodal Learning Reveals Plants' Hidden Sensory Integration Logic

# CoMM-BIP: Multimodal Integration with Biological Priors

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16281076.svg)](https://doi.org/10.5281/zenodo.16281076)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
<img src="docs/overview.png" width="600" alt="Model architecture">
</div>
Predict plant-microbe effectors by integrating transcriptomic, metabolomic, and phenomic data with biologically informed machine learning.

## Features
- **Biologically guided attention**: Prioritizes known gene-metabolite interactions.
- **State-of-the-art performance**: 0.98 F1 score on effector prediction.
- **Interpretable**: SHAP analysis and attention visualization.

## Quick Start
```bash
git clone https://github.com/MorillaLab/CoMM-BIP.git
pip install -r requirements.txt
python src/train.py --data_dir ./data
```

<!-- ============================================== -->
<div align="left">
  <h1 id="citation">🎈 Citation</h1>
  <hr style="height: 3px; background: linear-gradient(90deg, #EF8E8D, #5755A3); border: none; border-radius: 3px;">
</div>

If you find CoMM-BIP model helpful, please cite us.

```bibtex
@misc{vomo_donfack_2025_16281076,
  author       = {VOMO DONFACK, Kelly Larissa and
                  González Doblas, Verónica and
                  Morilla, Ian},
  title        = {Multimodal Learning Reveals Plants' Hidden Sensory
                   Integration Logic
                  },
  month        = jul,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0},
  doi          = {10.5281/zenodo.16281076},
  url          = {https://doi.org/10.5281/zenodo.16281076},
}
}
```
