import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--first", type=str)
parser.add_argument("--second", type=str)
args = parser.parse_args()

old_file  = open(args.first)
new_file  = open(args.second)

old_lines = old_file.readlines()
new_lines = new_file.readlines()
old_better = 0
new_better = 0

for old_line, new_line in zip(old_lines[:-1], new_lines[:-1]):
    old_file_name = old_line.split(" ")[0]
    new_file_name = new_line.split(" ")[0]
    # print(old_file_name)
    old_sad = old_line.split(" ")[1]
    new_sad = new_line.split(" ")[1]
    old_mse = old_line.split(" ")[2].split(":")[1]
    new_mse = new_line.split(" ")[2].split(":")[1]
    old_mse_float = float(old_mse)
    new_mse_float = float(new_mse)
    if old_mse_float > new_mse_float:
        print(str(old_mse_float) + " > " + str(new_mse_float))
        new_better += 1
    elif old_mse_float < new_mse_float:
        print(str(old_mse_float) + " < " + str(new_mse_float))
        old_better += 1
    else:
        print(str(old_mse_float) + " = " + str(new_mse_float))

print(str(old_better) + " " + str(new_better)) 