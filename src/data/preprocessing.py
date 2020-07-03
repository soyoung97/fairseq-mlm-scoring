import os, re, string
import numpy as np
from sklearn.model_selection import train_test_split
import subprocess

from tokenizer import SentencePieceTokenizer
from loader import CorpusLoader


class PreProcessing(): 

    def __init__(self, args):
        super().__init__()
        self.parallel_corpus = None
        self.args = args
        if args['tokenize']: 
            self.st = SentencePieceTokenizer(args)
            self.st.load_tokenizer()


    def clean_corpus_for_consistency(self, parallel_corpus):
        cleaned_parallel_corpus = []
        for src, tgt in parallel_corpus:
            # add space before/after punctuations
            src = re.sub(r'([{}])'.format(string.punctuation),r' \1 ', src)
            tgt = re.sub(r'([{}])'.format(string.punctuation),r' \1 ', tgt)
            cleaned_parallel_corpus.append((src, tgt))
        return parallel_corpus


    def split_train_test_valid(self, parallel_corpus):
        train_ratio, valid_ratio, test_ratio = self.args['train_test_valid_ratio']
        train_set, valid_and_test_set = train_test_split(
            parallel_corpus, 
            test_size = valid_ratio + test_ratio,
            random_state = self.args['seed'])
        valid_set, test_set = train_test_split(
            valid_and_test_set, 
            test_size = test_ratio / (valid_ratio + test_ratio),
            random_state = self.args['seed'])
        return train_set, valid_set, test_set


    def split_src_tgt(self, dataset):
        sources = []
        targets = []
        for pair in dataset:
            src, tgt = pair
            sources.append(src)
            targets.append(tgt)
        return sources, targets


    def save(self, dataset, flag="", src_tgt=""):
        data_type = ".".join([flag, src_tgt]).strip(".")
        if data_type == "": data_type = self.args['corpus_name']
        save_path = self.args['save_dir'] + \
                    self.args['corpus_name'] + "/" + \
                    data_type

        if os.path.exists(save_path):
            print("File already exists", save_path)
            return

        with open(save_path, 'w', encoding='utf8') as f:
            for data in dataset:
                if not self.args['split_src_target']:
                    data = "\t".join(data)
                f.write(data.strip() + '\n')


    def tokenize_corpus(self, parallel_corpus):
        parallel_corpus_tokenized = []
        for src, tgt in parallel_corpus:
            src_tok = self.st.tokenize(src, return_id=self.args['tokenize_to_id'])
            src_tok = " ".join(src_tok)
            tgt_tok = self.st.tokenize(tgt, return_id=self.args['tokenize_to_id'])
            tgt_tok = " ".join(tgt_tok)
            parallel_corpus_tokenized.append((src_tok, tgt_tok))
        return parallel_corpus_tokenized


    def preprocess(self, parallel_corpus):
        assert sum([len(pair) != 2 for pair in parallel_corpus]) == 0

        parallel_corpus = self.clean_corpus_for_consistency(parallel_corpus)
        print(len(parallel_corpus), len(set(parallel_corpus)))
        parallel_corpus = list(set(parallel_corpus))
        assert len(parallel_corpus) == len(set(parallel_corpus)) # checking duplicated pairs

        if self.args['tokenize']:
            parallel_corpus = self.tokenize_corpus(parallel_corpus)

        if self.args['split_src_target']:
            if self.args['split_train_test_valid']:
                train_set, valid_set, test_set = self.split_train_test_valid(parallel_corpus)
                print("Train set size:", len(train_set), train_set[0]); 
                print("Valid set size:", len(valid_set), valid_set[0]); 
                print("Test set size:", len(test_set), test_set[0])
                datasets = [train_set, valid_set, test_set]
                for data, flag in zip(datasets, ['train', 'test', 'valid']):
                    data_src, data_tgt = self.split_src_tgt(data)
                    self.save(data_src, flag=flag, src_tgt="src")
                    self.save(data_tgt, flag=flag, src_tgt="tgt")
            else:
                data_src, data_tgt = self.split_src_tgt(parallel_corpus)
                self.save(data_src, src_tgt="src")
                self.save(data_tgt, src_tgt="tgt")  
        else:
            if self.args['split_train_test_valid']:
                train_set, valid_set, test_set = self.split_train_test_valid(parallel_corpus)
                datasets = [train_set, valid_set, test_set]
                for data, flag in zip(datasets, ['train', 'test', 'valid']):
                    self.save(data, flag=flag)
            else:
                self.save(parallel_corpus)
        
        if self.args['fairseq-preprocess'] and self.args['split_src_target'] and self.args['split_train_test_valid']:
            self.binarize()


    def binarize(self):
        print("Running fairseq-preprocess...")

        preprocessed_dir = self.args['save_dir'] + self.args['corpus_name']    
        binarized_dir = self.args['binarized_dir'] + self.args['corpus_name'] + "/"
        vocab_name = self.args['spm_level'] + "_" + str(self.args['spm_vocab_size']) + ".fairseq.vocab"
        vocab_path = self.args['spm_vocab_dir'] + "/" + vocab_name

        command = ["fairseq-preprocess",
            "--source-lang", "src",
            "--target-lang", "tgt",
            "--trainpref", preprocessed_dir + "/train",
            "--validpref", preprocessed_dir + "/valid",
            "--testpref", preprocessed_dir + "/test",
            "--destdir", binarized_dir,
            "--srcdict", vocab_path,
            "--joined-dictionary"
        ]

        process = subprocess.run(command, check=True) # Removed text=True becuase of version issues
        if process.stdout is not None: print(process.stdout)

        print("Running fairseq-preprocess... Done.")


    def read_binarized_dataset(self, file_name, dataset_impl='mmap', max_line=10):
        from fairseq.data import data_utils, Dictionary, indexed_dataset
        
        input_path = self.args['binarized_dir'] + self.args['corpus_name'] + "/" + file_name
        vocab_path = self.args['binarized_dir'] + self.args['corpus_name'] + "/" + "dict.src.txt"

        dictionary = Dictionary.load(vocab_path)

        dataset = data_utils.load_indexed_dataset(
            input_path,
            dictionary,
            dataset_impl=dataset_impl,
            default='mmap'
        )

        for i, tensor_line in enumerate(dataset):
            if dictionary is None:
                line = ' '.join([str(int(x)) for x in tensor_line])
            else:
                line = dictionary.string(tensor_line)

            print(line)            
            if i == max_line: break



def main():

    args = {
        # loader args
        'corpus_name': "union",
        'read_src_tgt_only': True,

        # tokenizer args
        'spm_level': 'jamo',
        'spm_vocab_dir': './../../data/vocab/',
        'spm_vocab_size': 10000,
        'spm_coverage': 0.9995,

        # preprocessing args
        'save_dir' : './../../data/preprocessed_jamo/',
        'binarized_dir' : './../../data/binarized_jamo/',
        'split_src_target': True,
        'split_train_test_valid': True,
        'train_test_valid_ratio': (0.7, 0.15, 0.15),
        'seed': 0,
        'tokenize': True,
        'tokenize_to_id': False,
        'fairseq-preprocess': True
    }

    # pre-processing for parallel corpus : [(src, tgt), (src, tgt), ... , (src, tgt)]
    parallel_corpus = CorpusLoader(args).load_corpus()
    
    # should run convert_vocab.sh before preprocessing
    PreProcessing(args).preprocess(parallel_corpus)
    
    # explore binarized dataset
    PreProcessing(args).read_binarized_dataset("test.src-tgt.src")

if __name__ == "__main__":
    main()
