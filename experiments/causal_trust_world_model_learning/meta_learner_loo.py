"""Leave-one-out validation of the meta-learner's stated 2-feature rule.

Data: real per-backbone best trust method from tab:ablation_trust.
Features (as claimed in the paper): F1 = has built-in uncertainty
(RSSM-style recurrent state-space core), F2 = operates in latent space.
"""

BACKBONES = {
    "MLP":       {"F1": 0, "F2": 0, "best": "MultiStep"},
    "RSSM":      {"F1": 1, "F2": 0, "best": "None"},
    "JEPA":      {"F1": 0, "F2": 1, "best": "None"},
    "DreamerV3": {"F1": 1, "F2": 0, "best": "MultiStep"},  # DreamerV3 has an RSSM-style core -> F1=1
    "Diffusion": {"F1": 0, "F2": 0, "best": "EMA"},
}


def predict(held_out: str, train: dict) -> str:
    f1, f2 = BACKBONES[held_out]["F1"], BACKBONES[held_out]["F2"]
    # Explicit rule branch (not data-fit): latent space -> None/decoder-aware.
    if f2 == 1:
        return "None"
    # Otherwise: nearest neighbor by feature match among training backbones.
    matches = [name for name, v in train.items() if v["F1"] == f1 and v["F2"] == f2]
    if matches:
        return train[matches[0]]["best"]
    return "Ensemble"  # paper's stated fallback rule


def main() -> None:
    correct = 0
    for held_out in BACKBONES:
        train = {k: v for k, v in BACKBONES.items() if k != held_out}
        pred = predict(held_out, train)
        actual = BACKBONES[held_out]["best"]
        ok = pred == actual
        correct += ok
        print(f"held out {held_out:10s} features=({BACKBONES[held_out]['F1']},{BACKBONES[held_out]['F2']}) "
              f"predicted={pred:10s} actual={actual:10s} {'OK' if ok else 'WRONG'}")
    print(f"\nLeave-one-out accuracy: {correct}/{len(BACKBONES)} = {100*correct/len(BACKBONES):.0f}%")


if __name__ == "__main__":
    main()
