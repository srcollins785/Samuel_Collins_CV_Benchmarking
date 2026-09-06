"""Samuel Collins CV Benchmarking.

Compare classical machine-learning and neural-network image classifiers
through a single public function.
"""

from .benchmark import benchmark_image_classification

__all__ = ["benchmark_image_classification"]
__version__ = "0.1.0"
