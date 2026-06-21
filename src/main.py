import argparse
import pandas as pd
from src.pipeline.trainer import train_for_league


def main():
    parser = argparse.ArgumentParser(description="Early Game Winner Prediction")
    parser.add_argument('--league', type=str, default=None,
                        help='League name (e.g. LPL, LCK). Default: all leagues.')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Custom run name for MLflow.')
    parser.add_argument('--no-mlflow', action='store_true',
                        help='Disable MLflow logging.')
    args = parser.parse_args()

    results = train_for_league(
        league=args.league,
        run_name=args.run_name,
        use_mlflow=not args.no_mlflow
    )

    rows = []
    for model_name, res in results.items():
        row = {'model': model_name, 'best_params': str(res['best_params'])}
        row.update(res['metrics'])
        rows.append(row)

    df_result = pd.DataFrame(rows)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(df_result.to_string(index=False))
    print("=" * 60)


if __name__ == '__main__':
    main()
