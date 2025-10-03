import io
import os
import unicodedata
import string
import glob

import torch
import random

ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)
PATH = "data/names/*.txt"

def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

def load_data():
    data = {}
    def find_files(path): return glob.glob(path)
    def read_lines(filename):
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]
    for file_name in find_files(PATH):
        country = os.path.splitext(os.path.basename(file_name))[0]
        data[country] = read_lines(file_name)
    return data

def letter_to_index(letter): return ALL_LETTERS.find(letter)

def name_to_tensors(name):
    tensors = torch.zeros(len(name),1,N_LETTERS)
    for i,let in enumerate(name): tensors[i][0][letter_to_index(let)] = 1
    return tensors

def random_training_example(data):
    n_countries = len(data.keys())
    countries = sorted(list(data.keys()))
    rand_count = random.randint(0,n_countries-1)
    num_names = len(data[countries[rand_count]])
    names = data[countries[rand_count]]
    rand_name = random.randint(0,num_names-1)
    country_tensor = torch.tensor([rand_count]).long()
    return countries[rand_count], names[rand_name], country_tensor, name_to_tensors(names[rand_name])

if __name__ == "__main__":
    data = load_data()
    print(random_training_example(data))