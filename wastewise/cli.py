import argparse
import json
from pathlib import Path
from .inference import Predictor, decode_image


def main():
    p = argparse.ArgumentParser(description='Classify a local waste image')
    p.add_argument('image', type=Path)
    args = p.parse_args()
    result = Predictor().predict(decode_image(args.image.read_bytes()))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
