"""
Standalone training script entrypoint for running: python train.py
"""

from main import handle_train_command

if __name__ == "__main__":
    handle_train_command("config/config.yaml")
