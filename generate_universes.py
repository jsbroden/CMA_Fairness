# generate_universes.py
import itertools
import yaml
import pathlib

MODELS = ["logreg", "penalized_logreg", "rf"]
FEATURE_SETS = ["with_protected", "without_protected"]
THRESHOLDS = ["top15", "top30"]
EXCLUDE_GROUPS = ["keep-all", "drop-non-german"]


def build_universes():
    grid = list(itertools.product(MODELS, FEATURE_SETS, THRESHOLDS, EXCLUDE_GROUPS))

    # fresh list every call
    universes = [
        dict(
            id=i + 1,
            model=m,
            feature_set=fs,
            exclude_subgroups=excl,
            threshold_policy=thr,
        )
        for i, (m, fs, thr, excl) in enumerate(grid)
    ]
    return universes


def main(out_file="universes.yaml"):
    universes = build_universes()

    # --- safety: assert no duplicate tuples -------------------------
    assert len(universes) == len(
        {
            (
                u["model"],
                u["feature_set"],
                u["exclude_subgroups"],
                u["threshold_policy"],
            )
            for u in universes
        }
    ), "Duplicate design point detected!"

    # --- write / overwrite YAML -------------------------------------
    pathlib.Path(out_file).write_text(yaml.dump(universes, sort_keys=False))

    print(
        f"Generated {len(universes)} universes "
        f"({len(MODELS)}×{len(FEATURE_SETS)}×"
        f"{len(EXCLUDE_GROUPS)}×{len(THRESHOLDS)}). "
        f"Wrote → {out_file}"
    )


if __name__ == "__main__":
    main()
