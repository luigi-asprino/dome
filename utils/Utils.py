import pickle
import pandas as pd
import os

def load_list_from_file(filename, token_number, extractid=False):
    if (os.path.exists(filename + ".p")):
        return pickle.load(open(filename + ".p", "rb"))
    df = pd.read_csv(filename, sep='\t', header=None, usecols=[0])
    if extractid:
        result = [row[0].split("/")[token_number] for index, row in df.iterrows()]
    else:
        result = [row[0] for index, row in df.iterrows()]
    pickle.dump(result, open(filename + ".p", "wb"))
    return result

def load_map_from_file(filename):
    if (os.path.exists(filename + ".p")):
        return pickle.load(open(filename + ".p", "rb"))
    df = pd.read_csv(filename, sep='\t', header=None, usecols=[0, 1])
    map = {row[0]: row[1] for index, row in df.iterrows()}
    pickle.dump(map, open(filename + ".p", "wb"))
    return map


def load_matrix_from_file(filename, loadscore=False, exclude_null=False, lowercase=False):
    if (os.path.exists(filename + ".p")):
        return pickle.load(open(filename + ".p", "rb"))

    # print(f"Rows {rows} Cols {cols}")
    matrix = {}
    if loadscore:
        df = pd.read_csv(filename, sep='\t', header=None, usecols=[0, 1, 2])
    else:
        df = pd.read_csv(filename, sep='\t', header=None, usecols=[0, 1])
    for index, row in df.iterrows():
        if exclude_null and row[1] == "null":
            print("null" + row[0])
            continue
        k = row[0]
        if lowercase:
            if isinstance(k, str):
                k = k.lower()

        if k in matrix:
            if loadscore:
                matrix[k].append((row[1], float(row[2])))
            else:
                matrix[k].append((row[1], 1.0))
        else:
            if loadscore:
                matrix[k] = [(row[1], float(row[2]))]
            else:
                matrix[k] = [(row[1], 1.0)]

    pickle.dump(matrix, open(filename + ".p", "wb"))

    return matrix