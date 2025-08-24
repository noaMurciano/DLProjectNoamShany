"""
Personal Food Taste Classification

A deep learning package for learning personal food taste preferences
using transfer learning with EfficientNet-B3.

Author: Noam & Shany
Course: 046211 Deep Learning, Technion
"""

__version__ = "1.0.0"
__author__ = "Noam & Shany"
__course__ = "046211 Deep Learning"

from .model import Food101TasteClassifier, create_model
from .dataset import PersonalTasteDataset
from .train import PersonalFineTuner3Way
from .evaluate import evaluate_model

__all__ = [
    "Food101TasteClassifier",
    "create_model", 
    "PersonalTasteDataset",
    "PersonalFineTuner3Way",
    "evaluate_model"
]
