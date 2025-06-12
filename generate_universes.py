import itertools
import yaml

models = ["logreg", "penalized_logreg", "rf"]
feature_sets = ["with_protected", "without_protected"]
thresholds = ["top15", "top30", "middle30"]

universe_grid = list(itertools.product(models, feature_sets, thresholds))

universes = []
for i, (model, features, threshold) in enumerate(universe_grid):
    universes.append(
        {
            "id": i + 1,
            "model": model,
            "feature_set": features,
            "threshold_policy": threshold,
        }
    )

# Save to YAML
with open("universes.yaml", "w") as f:
    yaml.dump(universes, f)
