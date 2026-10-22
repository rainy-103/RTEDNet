"""RTED-Net implementation and checkpoint-compatible class names."""

from .network import TFGNetV9, TFGNetV101

RTEDNet = TFGNetV101

__all__ = ["RTEDNet", "TFGNetV9", "TFGNetV101"]
