import argparse
import json


def main(args) -> None:
    with open(f'{args.data_dir}/ETHOS_Binary/ETHOS_Binary_preprocessed.jsonl') as fin:
        fout_train = open(f'{args.data_dir}/ETHOS_Binary/ETHOS_Binary_preprocessed_train.jsonl', 'w')
        fout_test = open(f'{args.data_dir}/ETHOS_Binary/ETHOS_Binary_preprocessed_test.jsonl', 'w')
        for line in fin:
            split = json.loads(line.strip('\n'))['split']
            if split == 'train':
                fout_train.write(line)
            elif split == 'dev':
                fout_test.write(line)
            elif split == 'test':
                fout_test.write(line)
            else:
                raise Exception(f'Unexpected split: {split}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', help='Path to data directory.')
    cmd_args = parser.parse_args()
    main(cmd_args)
