import argparse
from ltv.pipeline.predict import PredictPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    PredictPipeline(clean=args.clean).run()
