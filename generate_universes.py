import itertools
import yaml

models = ["logreg", "penalized_logreg", "rf"]
feature_sets = ["with_protected", "without_protected"]
exclude_groups = ["keep-all", "drop-non-german"]
thresholds = ["top15", "top30"]  # "middle30"

universe_grid = list(
    itertools.product(models, feature_sets, exclude_groups, thresholds)
)

universes = []
for i, (model, feat_set, excl, thr) in enumerate(universe_grid, start=1):
    universes.append(
        dict(
            id=i,
            model=model,
            feature_set=feat_set,
            exclude_subgroups=excl,
            threshold_policy=thr,
        )
    )

# Save to YAML
with open("universes.yaml", "w") as f:
    yaml.dump(universes, f, sort_keys=False)

print(
    f"Generated {len(universes)} universes "
    f"({len(models)}×{len(feature_sets)}×{len(exclude_groups)}x{len(thresholds)})."
)
