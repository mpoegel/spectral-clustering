#! /bin/python

def data_to_csv(filename):
  filepath = 'data/raw/' + filename
  out_filepath = 'data/processed/' + filename + '.csv'
  in_fp = open(filepath, 'r')
  out_fp = open(out_filepath, 'w+')
  for line in in_fp:
    bits = line.strip().split(' ')
    bits = [ b[b.find(':') + 1:] if b.count(':') > 0 else b for b in bits ]
    bits.append(bits.pop(0))
    newline = ','.join(bits)
    out_fp.write(newline + '\n')
  in_fp.close()
  out_fp.close()


if __name__ == '__main__':
  print('=> Preprocessing data...')
  print('. processing usps')
  data_to_csv('usps')
  print('. processing mnist')
  data_to_csv('mnist')
  print('=> Preprocessing completed.')
