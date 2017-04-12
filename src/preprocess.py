#! /bin/python

def data_to_csv(filename, dim):
    filepath = 'data/raw/' + filename
    out_filepath = 'data/processed/' + filename + '.csv'
    in_fp = open(filepath, 'r')
    out_fp = open(out_filepath, 'w+')
    for line in in_fp:
        bits = line.strip().split(' ')
        i = 1
        k = 1
        full_bits = [bits[0]]
        while i < dim and k < len(bits):
            n, b = bits[k].split(':')
            if int(n) == i:
                full_bits.append(b)
                k += 1
            else:
                full_bits.append('0')
            i += 1
        while len(full_bits) < dim:
            full_bits.append('0')
        full_bits.append(full_bits.pop(0))
        newline = ','.join(full_bits)
        out_fp.write(newline + '\n')
    in_fp.close()
    out_fp.close()


if __name__ == '__main__':
    print('=> Preprocessing data...')
    print('. processing usps')
    data_to_csv('usps', 256)
    print('. processing mnist')
    data_to_csv('mnist', 784)
    print('=> Preprocessing completed.')
