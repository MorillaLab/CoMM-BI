#!/usr/bin/env python3
"""
CoMM-BIP Data Loader
Handles loading and preprocessing of:
- Transcriptomics (RNA-seq)
- Metabolomics (LC-MS)
- Phenomics (imaging features)
- Environmental data (genotype/effector)
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

class MultimodalDataset(Dataset):
    """PyTorch Dataset for biological multimodal data"""
    
    def __init__(self, 
                 rna: np.ndarray,
                 metabo: np.ndarray,
                 pheno: np.ndarray,
                 env: np.ndarray,
                 labels: np.ndarray,
                 augment: bool = False,
                 noise_level: float = 0.05):
        """
        Args:
            rna: Transcriptomic data (n_samples x n_genes)
            metabo: Metabolomic data (n_samples x n_metabolites)
            pheno: Phenotypic features (n_samples x n_traits)
            env: Environmental variables (n_samples x 1)
            labels: Effector classes (n_samples)
            augment: Enable data augmentation
            noise_level: Gaussian noise SD for augmentation
        """
        self.rna = torch.FloatTensor(rna)
        self.metabo = torch.FloatTensor(metabo)
        self.pheno = torch.FloatTensor(pheno)
        self.env = torch.FloatTensor(env)
        self.labels = torch.LongTensor(labels)
        self.augment = augment
        self.noise_level = noise_level
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        if self.augment and torch.rand(1) > 0.7:  # 30% augmentation chance
            return self._augment_sample(idx)
        return (
            self.rna[idx],
            self.metabo[idx],
            self.pheno[idx],
            self.env[idx],
            self.labels[idx]
        )
    
    def _augment_sample(self, idx):
        """Apply Gaussian noise augmentation"""
        noise = lambda x: x + torch.randn_like(x) * self.noise_level
        return (
            noise(self.rna[idx]),
            noise(self.metabo[idx]),
            noise(self.pheno[idx]),
            self.env[idx],  # Don't augment environment
            self.labels[idx]
        )

class BiologicalDataLoader:
    """Main data loading and preprocessing class"""
    
    def __init__(self, 
                 data_dir: str = "data",
                 batch_size: int = 64,
                 test_size: float = 0.2,
                 val_size: float = 0.1):
        """
        Args:
            data_dir: Directory containing TSV files
            batch_size: Dataloader batch size
            test_size: Fraction of data for testing
            val_size: Fraction of training data for validation
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.test_size = test_size
        self.val_size = val_size
        self.scalers = {}
        self.feature_names = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load raw data from TSV files"""
        try:
            return {
                'transcriptome': pd.read_csv(os.path.join(self.data_dir, "transcriptome.tsv"), 
                                           sep="\t", index_col=0).T,
                'metabolomics': pd.read_csv(os.path.join(self.data_dir, "metabolomics.tsv"), 
                                          sep="\t", index_col=0),
                'phenomics': pd.read_csv(os.path.join(self.data_dir, "phenomics.tsv"), 
                                       sep="\t", index_col=0),
                'environment': pd.read_csv(os.path.join(self.data_dir, "environment.tsv"), 
                                         sep="\t", index_col=0)
            }
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return self._generate_dummy_data()
    
    def _generate_dummy_data(self) -> Dict[str, pd.DataFrame]:
        """Generate synthetic data for testing"""
        n_samples = 200
        samples = [f"Sample_{i}" for i in range(n_samples)]
        effector = np.random.randint(0, 4, size=n_samples)
        genotype = np.random.choice([0, 1], size=n_samples)
        
        return {
            'transcriptome': pd.DataFrame(
                np.random.randn(n_samples, 1000) + effector[:, None] * 0.8,
                index=samples,
                columns=[f"Gene_{i}" for i in range(1000)]
            ),
            'metabolomics': pd.DataFrame(
                np.random.randn(n_samples, 50) + genotype[:, None] * 0.5,
                index=samples,
                columns=[f"Metab_{i}" for i in range(50)]
            ),
            'phenomics': pd.DataFrame(
                np.random.randn(n_samples, 3) + effector[:, None] * 0.3,
                index=samples,
                columns=['Root_length', 'Lateral_count', 'Fractal_dim']
            ),
            'environment': pd.DataFrame({
                'Genotype': genotype,
                'Effector': effector
            }, index=samples)
        }
    
    def preprocess_data(self, data: Dict[str, pd.DataFrame]) -> Tuple:
        """Clean and normalize multimodal data"""
        # 1. Handle missing values
        dfs = {}
        for modality, df in data.items():
            df = df.apply(pd.to_numeric, errors='coerce')
            df = df.loc[:, (df.isna().mean() < 0.3)]  # Drop >30% NA cols
            dfs[modality] = df.fillna(df.median())
        
        # 2. Get common samples
        common_samples = set(dfs['transcriptome'].index)
        for df in dfs.values():
            common_samples.intersection_update(df.index)
        common_samples = list(common_samples)
        
        # 3. Scale features
        scaled = {}
        for modality, df in dfs.items():
            if modality == 'environment':
                scaled[modality] = df.loc[common_samples]
                continue
                
            self.scalers[modality] = StandardScaler()
            scaled_df = self.scalers[modality].fit_transform(df.loc[common_samples])
            scaled[modality] = pd.DataFrame(
                scaled_df,
                index=common_samples,
                columns=df.columns
            )
            self.feature_names[modality] = df.columns.tolist()
        
        # 4. Prepare targets
        env = scaled['environment']
        env['Genotype'] = env['Genotype'].astype(int)  # Ensure binary
        labels = env['Effector'].values
        
        return (
            scaled['transcriptome'].values,
            scaled['metabolomics'].values,
            scaled['phenomics'].values,
            env[['Genotype']].values,
            labels
        )
    
    def get_dataloaders(self, augment_train: bool = True) -> Dict[str, DataLoader]:
        """Generate train/val/test dataloaders"""
        raw_data = self.load_data()
        rna, metabo, pheno, env, labels = self.preprocess_data(raw_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            np.arange(len(labels)), labels,
            test_size=self.test_size,
            random_state=SEED,
            stratify=labels
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.val_size,
            random_state=SEED,
            stratify=y_train
        )
        
        # Create datasets
        datasets = {
            'train': MultimodalDataset(
                rna[X_train], metabo[X_train], pheno[X_train], 
                env[X_train], y_train, augment=augment_train
            ),
            'val': MultimodalDataset(
                rna[X_val], metabo[X_val], pheno[X_val],
                env[X_val], y_val
            ),
            'test': MultimodalDataset(
                rna[X_test], metabo[X_test], pheno[X_test],
                env[X_test], y_test
            )
        }
        
        # Create dataloaders
        return {
            'train': DataLoader(
                datasets['train'],
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=min(4, os.cpu_count())
            ),
            'val': DataLoader(
                datasets['val'],
                batch_size=self.batch_size,
                num_workers=min(4, os.cpu_count())
            ),
            'test': DataLoader(
                datasets['test'],
                batch_size=self.batch_size,
                num_workers=min(4, os.cpu_count())
            )
        }

# Example usage
if __name__ == "__main__":
    loader = BiologicalDataLoader(data_dir="data")
    dataloaders = loader.get_dataloaders()
    
    print("\nData loading complete!")
    print(f"Train batches: {len(dataloaders['train']}")
    print(f"Feature counts: { {k: len(v) for k, v in loader.feature_names.items()} }")
