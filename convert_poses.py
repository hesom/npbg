import argparse

parser = argparse.ArgumentParser(description="Converts Redwood pose format to npbg format")
parser.add_argument("inpath", type=str, help="path to input file")
parser.add_argument("outpath", type=str, help="path to output file")

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.inpath, "r") as f, open(args.outpath, "w") as outF:
        for i, line in enumerate(f):
            if i%5 != 0:
                outF.write(line)
