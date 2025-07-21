
    LearningRateMonitor
)
from pytorch_lightning.loggers import CSVLogger
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# Local imports
from model import (  # Assuming your model code is in model.py
    CoMM_BIP,
    PlantDataModule,
    DiagnosticCallback,
    set_seed,
    SEED
)

# Configuration
CONFIG = {
    "batch_size": 64,
    "max_epochs": 100,
    "lr": 2e-5,
    "weight_decay": 1e-4,
    "patience": 10,
    "seed": SEED,
    "data_paths": {
        "transcriptome": "data/transcriptome.tsv",
        "metabolomics": "data/metabolomics.tsv",
        "phenomics": "data/phenomics.tsv",
        "environment": "data/environment.tsv"
    },
    "output_dir": "results"
}

def setup_directories(output_dir: str) -> None:
    """Create output directories if they don't exist"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)

def train_model() -> Dict:
    """Main training pipeline"""
    set_seed(CONFIG["seed"])
    setup_directories(CONFIG["output_dir"])

    # Initialize data module
    dm = PlantDataModule(
        batch_size=CONFIG["batch_size"],
        num_workers=min(4, os.cpu_count())
    )
    
    try:
        dm.prepare_data()
        dm.setup()
        print("? Data loaded successfully")
    except Exception as e:
        print(f"?? Using dummy data: {str(e)}")
        dm._create_dummy_data()
        dm.setup()

    # Initialize model
    model = CoMM_BIP(
        input_dims={
            'rna': len(dm.feature_names['rna']),
            'metabo': len(dm.feature_names['metabo']),
            'pheno': len(dm.feature_names['pheno']),
            'env': 1
        },
        feature_names=dm.feature_names,
        temperature=0.1
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_acc",
            patience=CONFIG["patience"],
            mode="max",
            verbose=True
        ),
        ModelCheckpoint(
            dirpath=os.path.join(CONFIG["output_dir"], "checkpoints"),
            filename="best_model",
            monitor="val_acc",
            mode="max",
            save_top_k=1
        ),
        LearningRateMonitor(),
        DiagnosticCallback()
    ]

    # Logger
    logger = CSVLogger(
        save_dir=os.path.join(CONFIG["output_dir"], "logs"),
        name="comm_bip"
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=CONFIG["max_epochs"],
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        enable_progress_bar=True,
        enable_model_summary=True,
        precision="bf16-mixed",
        accelerator="auto",
        devices="auto"
    )

    # Training
    print("\n?? Starting training...")
    trainer.fit(model, dm)

    # Testing
    print("\n?? Running final evaluation...")
    test_results = trainer.test(
        model, 
        dataloaders=dm.test_dataloader(),
        ckpt_path="best"
    )

    # Save final model
    best_model_path = os.path.join(
        CONFIG["output_dir"],
        "checkpoints",
        "final_model.pt"
    )
    torch.save(model.state_dict(), best_model_path)
    print(f"\n?? Model saved to {best_model_path}")

    return {
        "model": model,
        "datamodule": dm,
        "test_results": test_results
    }

if __name__ == "__main__":
    # Run training pipeline
    results = train_model()

    # Print final metrics
    print("\n?? Final Test Metrics:")
    for k, v in results["test_results"][0].items():
        print(f"{k}: {v:.4f}")

    # Save feature importance plot (example)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=list(results["model"].feature_importance().keys()),
        y=list(results["model"].feature_importance().values())
    )
    plt.title("Feature Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(CONFIG["output_dir"], "figures", "feature_importance.png"))
    print("\n?? Visualization saved to results/figures/")
