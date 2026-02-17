"""
CLI entry point — trains a model with Optuna + MLflow.

Usage:
    # Train a single model
    uv run python -m src.cli --model xgboost --trials 100
    uv run python -m src.cli --model ridge --trials 200 --generate-submission

    # Train all models
    uv run python -m src.cli --train-all --task regression
    uv run python -m src.cli --model all --task regression --trials 50
"""

import argparse

from src.application.services.training_service import TrainingService
from src.domain.entities.dataset import Dataset
from src.domain.entities.experiment_config import ExperimentConfig, TaskType
from src.infrastructure.adapters.data_loaders.csv_loader import CsvDataLoader
from src.infrastructure.adapters.trackers.mlflow_tracker import MLflowTracker
from src.infrastructure.config.settings import Settings

# Registry of available model adapters
MODEL_REGISTRY: dict[str, callable] = {}


def _register_models():
    """Lazily register available models."""

    # sklearn models (always available)
    from src.infrastructure.adapters.models.sklearn import (
        GradientBoostingAdapter, LinearRegressionAdapter, RandomForestAdapter,
        RidgeAdapter, SVMAdapter)

    MODEL_REGISTRY["linear"] = lambda task: LinearRegressionAdapter(task_type=task)
    MODEL_REGISTRY["ridge"] = lambda task: RidgeAdapter(task_type=task)
    MODEL_REGISTRY["random_forest"] = lambda task: RandomForestAdapter(task_type=task)
    MODEL_REGISTRY["gradient_boosting"] = lambda task: GradientBoostingAdapter(
        task_type=task
    )
    MODEL_REGISTRY["svm"] = lambda task: SVMAdapter(task_type=task)

    # XGBoost
    try:
        from src.infrastructure.adapters.models.xgboost import XGBoostAdapter

        MODEL_REGISTRY["xgboost"] = lambda task: XGBoostAdapter(task_type=task)
    except ImportError:
        pass

    # PyTorch
    try:
        from src.infrastructure.adapters.models.pytorch import PyTorchAdapter

        MODEL_REGISTRY["pytorch"] = lambda task: PyTorchAdapter(task_type=task)
    except ImportError:
        pass

    # TensorFlow
    try:
        from src.infrastructure.adapters.models.tensorflow import \
            TensorFlowAdapter

        MODEL_REGISTRY["tensorflow"] = lambda task: TensorFlowAdapter(
            task_type=task,
        )
    except ImportError:
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kaggle competition model trainer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        default=None,
        help=("Model to train (e.g., linear, ridge, xgboost, pytorch, all)"),
    )
    parser.add_argument(
        "--train-all",
        action="store_true",
        help="Train all available models",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="regression",
        choices=[
            "regression",
            "binary_classification",
            "multiclass_classification",
        ],
        help="Task type (default: regression)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of Optuna trials (default: auto-set based on model complexity)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--generate-submission",
        action="store_true",
        help="Generate submission file after training",
    )
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Preprocess raw data and save to processed directory",
    )
    return parser.parse_args()


def train_single_model(
    model_name: str,
    task_type: TaskType,
    n_trials: int | None,
    cv_folds: int,
    generate_submission: bool,
    settings: Settings,
    dataset: Dataset,
    test_df=None,
    train_df=None,
) -> tuple:
    """Train a single model and optionally generate submission."""
    model_adapter = MODEL_REGISTRY[model_name](task_type)

    # Determine number of trials
    if n_trials is not None:
        trials = n_trials
    else:
        trials = model_adapter.get_default_trials()

    # Configure experiment
    config = ExperimentConfig(
        experiment_name=(f"{settings.competition_name} - {model_adapter.name}"),
        model_name=model_adapter.name,
        task_type=task_type,
        n_trials=trials,
        cv_folds=cv_folds,
        random_state=settings.random_state,
    )

    # Train
    tracker = MLflowTracker(settings)
    service = TrainingService(model_adapter, tracker, config)

    print("\n" + "=" * 60)
    print(f"  {settings.competition_name.upper()} - {model_adapter.name}")
    print(f"  Trials: {trials} (model-optimized)")
    print("=" * 60)

    # Run training with model-specific submission file
    model, study, metrics = service.run(
        dataset=dataset,
        generate_submission=generate_submission,
        test_df=test_df,
        train_df=train_df,
        settings=settings,
        submission_filename=f"submission_{model_name}.csv",
    )

    return model, study, metrics


def train_all_models(
    task_type: TaskType,
    n_trials: int | None,
    cv_folds: int,
    settings: Settings,
    dataset: Dataset,
    test_df=None,
    train_df=None,
) -> dict:
    """Train all registered models and generate submissions for each."""
    import os

    import pandas as pd

    results = {}

    # Check if test data exists for submission generation
    train_path = os.path.join(settings.raw_data_dir, "train.csv")
    test_path = os.path.join(settings.raw_data_dir, "test.csv")

    if os.path.exists(test_path) and os.path.exists(train_path):
        if test_df is None or train_df is None:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
        generate_submission = True
        print("✓ Test data found - submissions will be generated for all models")
    else:
        generate_submission = False
        print(
            f"⚠️  Test data not found at {test_path} - no submissions will be generated"
        )

    print("\n" + "=" * 60)
    print(f"  TRAINING ALL MODELS ({len(MODEL_REGISTRY)} total)")
    print("=" * 60)

    for i, model_name in enumerate(sorted(MODEL_REGISTRY.keys()), 1):
        print(f"\n[{i}/{len(MODEL_REGISTRY)}] Training {model_name}...")
        try:
            model, study, metrics = train_single_model(
                model_name=model_name,
                task_type=task_type,
                n_trials=n_trials,
                cv_folds=cv_folds,
                generate_submission=generate_submission,
                settings=settings,
                dataset=dataset,
                test_df=test_df,
                train_df=train_df,
            )
            results[model_name] = {
                "model": model,
                "study": study,
                "metrics": metrics,
                "success": True,
            }
            print(f"✓ {model_name} completed successfully")
        except Exception as e:
            print(f"✗ {model_name} failed: {e}")
            results[model_name] = {"success": False, "error": str(e)}

    # Print summary
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)

    successful = [name for name, res in results.items() if res["success"]]
    failed = [name for name, res in results.items() if not res["success"]]

    print(f"\n✓ Successful: {len(successful)}/{len(MODEL_REGISTRY)}")
    if successful:
        for model_name in successful:
            metrics = results[model_name]["metrics"]
            metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
            print(f"  - {model_name}: {metric_str}")

    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(MODEL_REGISTRY)}")
        for model_name in failed:
            print(f"  - {model_name}: {results[model_name]['error']}")

    if generate_submission:
        print("\n📝 Submission files generated:")
        for model_name in successful:
            print(f"  - submission_{model_name}.csv")

    return results


def preprocess_data(settings: Settings) -> None:
    """Preprocess raw data and save to processed directory."""
    import os

    import pandas as pd

    from src.infrastructure.adapters.preprocessors.sklearn_preprocessor import \
        SklearnPreprocessor

    print("\n" + "=" * 60)
    print("  DATA PREPROCESSING")
    print("=" * 60)

    # Load raw data
    train_path = os.path.join(settings.raw_data_dir, "train.csv")
    if not os.path.exists(train_path):
        print(f"✗ Error: Training data not found at {train_path}")
        return

    print(f"\nLoading raw data from {train_path}...")
    train_df = pd.read_csv(train_path)
    print(f"✓ Loaded training data: {train_df.shape}")

    # Determine target column (adjust for your competition)
    target_col = "Transported"  # Default for Spaceship Titanic
    if target_col not in train_df.columns:
        print(f"✗ Error: Target column '{target_col}' not found in data")
        print(f"  Available columns: {', '.join(train_df.columns)}")
        return

    # Apply preprocessing and split
    preprocessor = SklearnPreprocessor()
    print("\nApplying preprocessing pipeline...")
    dataset = preprocessor.get_train_test_split(
        data=train_df,
        target_col=target_col,
        test_size=0.2,
        random_state=settings.random_state,
    )

    # Create processed directory if it doesn't exist
    os.makedirs(settings.processed_data_dir, exist_ok=True)

    # Save processed data
    output_dir = settings.processed_data_dir
    print(f"\nSaving processed data to {output_dir}...")

    dataset.X_train.to_parquet(os.path.join(output_dir, "X_train.parquet"))
    dataset.X_test.to_parquet(os.path.join(output_dir, "X_test.parquet"))
    pd.DataFrame(dataset.y_train).to_parquet(
        os.path.join(output_dir, "y_train.parquet")
    )
    pd.DataFrame(dataset.y_test).to_parquet(os.path.join(output_dir, "y_test.parquet"))

    print("\n✓ Preprocessing completed successfully:")
    print(f"  - X_train: {dataset.X_train.shape}")
    print(f"  - X_test: {dataset.X_test.shape}")
    print(f"  - y_train: {dataset.y_train.shape}")
    print(f"  - y_test: {dataset.y_test.shape}")
    print(f"\n  Features ({len(dataset.X_train.columns)}):")
    for col in sorted(dataset.X_train.columns):
        print(f"    - {col}")


def main():
    args = parse_args()

    _register_models()

    settings = Settings()

    # Handle preprocessing command
    if args.preprocess:
        preprocess_data(settings)
        return

    # Determine if training all models
    if args.train_all or (args.model and args.model.lower() == "all"):
        train_all = True
        model_name = None
    elif args.model:
        train_all = False
        model_name = args.model
        if model_name not in MODEL_REGISTRY:
            available = ", ".join(sorted(MODEL_REGISTRY.keys()))
            print(f"✗ Unknown model: '{model_name}'")
            print(f"  Available models: {available}")
            return
    else:
        print("✗ Error: Either --model or --train-all must be specified")
        print("  Use --model <model_name> to train a specific model")
        print("  Use --train-all to train all available models")
        return

    task_type = TaskType(args.task)

    # Load data
    loader = CsvDataLoader(
        raw_dir=settings.raw_data_dir,
        processed_dir=settings.processed_data_dir,
    )
    X_train, X_test, y_train, y_test = loader.load_processed()
    dataset = Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    # Load raw data if submission is requested
    test_df = None
    train_df = None
    if args.generate_submission or train_all:
        import os

        import pandas as pd

        train_path = os.path.join(settings.raw_data_dir, "train.csv")
        test_path = os.path.join(settings.raw_data_dir, "test.csv")

        if os.path.exists(test_path) and os.path.exists(train_path):
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            if not train_all:
                print("✓ Loaded raw data for submission generation")
        elif args.generate_submission:
            print(
                f"⚠️  Test data not found at {test_path}, "
                "submission will not be generated"
            )
            args.generate_submission = False

    # Train all models or single model
    if train_all:
        train_all_models(
            task_type=task_type,
            n_trials=args.trials,
            cv_folds=args.cv_folds,
            settings=settings,
            dataset=dataset,
            test_df=test_df,
            train_df=train_df,
        )
        print("\n" + "=" * 60)
        print("  ALL MODELS TRAINING COMPLETED")
        print("=" * 60)
    else:
        # Single model training
        model, study, metrics = train_single_model(
            model_name=model_name,
            task_type=task_type,
            n_trials=args.trials,
            cv_folds=args.cv_folds,
            generate_submission=args.generate_submission,
            settings=settings,
            dataset=dataset,
            test_df=test_df,
            train_df=train_df,
        )

        print("\n" + "=" * 60)
        print("  TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 60)


if __name__ == "__main__":
    main()
