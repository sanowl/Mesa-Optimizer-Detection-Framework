# Mesa-Optimizer Detection Framework (Research Prototype)

Welcome!  This documentation describes a *prototype* toolkit for exploring
possible signals of **mesa-optimization / deceptive alignment** in neural
networks.

> **Important**  
> This code is *experimental*.  No publicly-available model has yet been
> proven to contain a mesa-optimizer, so the framework is evaluated only on
> toy or synthetic cases.  **Do not depend on risk scores for high-stakes
> decisions.**

## Getting Started

1.  Install dev requirements and editable package:

    ```bash
    pip install -e .[dev]
    ```
2.  Run the quick demo:

    ```bash
    python examples/basic_usage.py
    ```
3.  Explore the toy benchmark:

    ```bash
    python benchmark_evaluation.py --quick
    ```

## Limitations

* No ground-truth dataset of real mesa-optimisers.  Scores are heuristic.
* Memory usage is tuned for small/medium models, not GPT-3-class networks.
* Heuristics can be evaded by an adaptive adversary.
* Calibration and Bayesian aggregation are TODO.

See the [Roadmap](../README.md#current-capabilities) in the README for the
work-in-progress plan. 