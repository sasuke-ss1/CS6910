from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--question", "-q", type=int, default=None, help="The Question Number you want to run")
args = parser.parse_args()

print(args)