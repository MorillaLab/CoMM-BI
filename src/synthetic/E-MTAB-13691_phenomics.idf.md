## High-Throughput Phenotyping of Effector-Expressing Tomato Roots

### Experimental Overview

| Metadata Field         | Details |
|-----------------------|---------|
| **Investigation Title** | High-throughput phenotyping of effector-expressing tomato (*Solanum lycopersicum*) roots |
| **Experimental Design** | Automated image analysis of T-DNA transformed hairy roots 24h post-induction with β-estradiol (10µM) |

### Synthetic Protocols

**1. Root Imaging**  
- **Instrument**: 5MP RGB camera (e.g., Canon EOS RP)  
- **Resolution**: 10μm/pixel  
- **Angles**: 3 views per sample (0°, 45°, 90°)  
- **Lighting**: White LED array (5000K, 2000 lux)  

**2. Image Analysis Pipeline**  
| Step | Tool/Parameters |
|------|-----------------|
| Primary Analysis | RhizoVision Explorer 2.0 ([RVE2](https://rhizovision.github.io)) <br> - Root tracing algorithm v3.1.4 <br> - Min. root diameter: 150μm |
| Secondary Analysis | Custom Python scripts ([GitHub](https://github.com/yourusername/root-phenotyping)) <br> - OpenCV 4.5 + scikit-image <br> - Morphometric features: 28 descriptors |

**Quality Controls**  
- Calibration: NIH ImageJ with micrometer scale bar  
- Negative Controls: Empty vector (pER8) lines  
- Replicates: 5 biological replicates × 3 technical replicates  

> **Sample Size**:  
> - 8 effector lines (RiSP749, GLOIN707, etc.)  
> - 15 plants per line  
> - Total *n* = 120 samples
