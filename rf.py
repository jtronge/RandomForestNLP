import csv

def load_csv(fname):
    with open(fname) as fp:
        csv_reader = csv.reader(fp)
        return [tuple(line) for line in csv_reader]


def main():
    labels = load_csv('OLID/labels-levela.csv')
    print(labels[:10])

    data = load_csv('OLID/testset-levela.tsv')
    print(data[:10])

if __name__ == '__main__':
    main()
