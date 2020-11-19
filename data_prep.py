def prepare_data(file):
    lines = []
    inputs = []
    group = []
    exp = []
    f = open(file, "r")
    for line in f:
        line = line.split()
        if len(line) > 2:
            inputs.append([float(line[0]), float(line[1])])
        else:
            inputs.append([float(line[0])])
        exp.append(float(line[len(line)-1]))
    print(inputs)
    print(exp)
    f.close()
    return inputs, exp
