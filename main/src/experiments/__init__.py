from src.experiments.notebook_api import (
    # core
    load_cfg, save_cfg, deep_update,
    build_datasets, build_loaders, build_model, build_trainer,
    make_optimizer, maybe_init_wandb, build_trainer_from_checkpoint,
    make_checkpoint_path, save_checkpoint, load_checkpoint,
    train_epochs, evaluate_generation, collect_features,
    run_linear_probe, run_experiment, load_and_evaluate,
    run_ablation_suite, run_timestep_sweep,
    # new experiments
    run_feature_layer_ablation, FEATURE_LAYERS,
    train_with_curves, compare_training_curves,
    collect_gate_histogram, run_gating_analysis,
    run_ema_momentum_sweep,
    generate_sample_grid, generate_comparison_grid,
    extract_features_for_viz, run_umap_comparison, run_tsne_comparison,
)
