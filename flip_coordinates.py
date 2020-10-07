import argparse

parser = argparse.ArgumentParser(description="Clips y and z coordinate of view matrices")
parser.add_argument("inpath", type=str, help="path to input file")
parser.add_argument("outpath", type=str, help="path to output file")

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.inpath, "r") as f, open(args.outpath, "w") as outF:
        for line in f:
            nums = [float(s) for s in line.split(" ") if isfloat(s)]
            
            # flip second and third column
            nums[2] *= -1

            newLine = " ".join([str(n) for n in nums]) + "\n"
            outF.write(newLine)
