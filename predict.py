"""
Standalone prediction script entrypoint for running: python predict.py
"""

import sys
from main import handle_predict_command, print_formatted_prediction
from src.prediction.predictor import TicketPredictor

if __name__ == "__main__":
    predictor = TicketPredictor(config_path="config/config.yaml")

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        res = predictor.predict_single(text)
        print_formatted_prediction(res)
    else:
        sample_tickets = [
            "Unable to login after password reset.",
            "I was charged twice for my subscription this month.",
            "Where can I submit my leave application and bank details update?",
            "What are your general support and office opening hours?",
            "Do you offer student or non-profit discounts on pricing?"
        ]
        print("\n" + "=" * 60)
        print("   RUNNING DEMO PREDICTIONS WITH CONFIDENCE SCORING   ")
        print("=" * 60)
        for t in sample_tickets:
            res = predictor.predict_single(t)
            print_formatted_prediction(res)
