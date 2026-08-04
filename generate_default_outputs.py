from pathlib import Path

from todim_family_engine import (
    METHOD_ORDER,
    built_in_example,
    compare_methods,
    sensitivity_analysis,
    summarize_sensitivity,
)


def main() -> None:
    output_dir = Path(__file__).resolve().parent
    data = built_in_example()
    comparison = compare_methods(
        data,
        methods=METHOD_ORDER,
        normalization="sum",
        reference_method="exptodim",
        theta=1.0,
        lambda_=2.25,
        rho=3.0,
        alpha=0.88,
        beta=0.88,
    )
    sensitivity = sensitivity_analysis(data, methods=METHOD_ORDER)
    summary = summarize_sensitivity(sensitivity)

    comparison.to_csv(output_dir / "method_comparison_default.csv", index=False)
    sensitivity.to_csv(output_dir / "sensitivity_results_todim_family_default.csv", index=False)
    summary.to_csv(output_dir / "robustness_summary_default.csv", index=False)

    print(f"Generated {len(comparison)} method-comparison rows")
    print(f"Generated {len(sensitivity)} sensitivity scenarios")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
