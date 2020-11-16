def prepare_data(file):
    lines = []
    inputs = []
    exp = []
    f = open(file, "r")
    for line in f:
        line = line.split(' ')
        for thing in line:
            if thing == '':
                continue
            else:
                lines.append(thing.rstrip('\n'))

    for item in range(len(lines)):
        if item % 2 == 0:
            exp.append(float(lines[item]))
        else:
            inputs.append(float(lines[item]))

    f.close()

    return inputs, exp
print(prepare_data("Data/1in_cubic.txt"))