from torch.utils.data import Dataset, DataLoader
import json
import torch.nn.functional as F
import copy
import argparse
import os
import torch
from fairseq.models.roberta import RobertaModel
from jiwer import wer
import tqdm


def get_args():
    parser = argparse.ArgumentParser(description='Preprocess librispeech json type dataset.')
    parser.add_argument('--ORIG_PATH', help='json file path')
    parser.add_argument('--MAX_CONV', default=3, help='maximum number of conversations(affects the total speed)')
    parser.add_argument('--PPPL_PATH', help='path to save the preprocessed file')
    parser.add_argument('--PPPL_SCORE', help='pppl scoring mode', action="store_true")
    parser.add_argument('--RESCORE', help='rescoring mode, giving wer', action="store_true")
    parser.add_argument('--SCALE', default=0.5, help='weight for when calculating rescoring')
    args = parser.parse_args()
    return args

def pppl_score(args):
    orig_dataset = JSONDataset(args.ORIG_PATH)
    scorer = MLMScorer()
    pppl_dataset = scorer.get_pppl_scored_dataset(orig_dataset)
    pppl_path = args.ORIG_PATH.split('.json')[0] + '.pppl.json'
    scorer.save_json(pppl_dataset, pppl_path)
    print(f"Saved new pppl scored dataset at {pppl_path}.")


def rescore(args):
    orig_dataset = JSONDataset(args.ORIG_PATH, max_conv=args.MAX_CONV)
    pppl_dataset = JSONDataset(args.PPPL_PATH, max_conv=args.MAX_CONV)
    print("Calculating WER for original dataset ... ")
    scorer = MLMScorer()
    scorer.calculate_wer(orig_dataset.json)
    rescore_path = args.ORIG_PATH.split('.json')[0] + '.rescored.json'
    rescored_dataset = scorer.rescore(orig_dataset.json, pppl_dataset.json, scale=args.SCALE)
    scorer.save_json(rescored_dataset, rescore_path)
    print(f"Saved rescored dataset at {rescore_path}.")
    scorer.calculate_wer(rescored_dataset)


def main():
    args = get_args()
    if args.PPPL_SCORE:
        pppl_score(args)
    elif args.RESCORE:
        rescore(args)
    
class JSONDataset(Dataset):
    
    def __init__(self, file_path, max_conv=3):
        super().__init__()
        self.file_path = file_path
        self.json = dict()
        self.raw_json = None
        self.uid_list = None
        self.max_conv = max_conv
        self.load_json()
        if 'other' in file_path:
            self.split = 100
        else:
            self.split = 99

    def load_json(self):
        with open(self.file_path) as f:
            self.raw_json = json.load(f)
            self.uid_list = list(self.raw_json.keys())
        if len(self.uid_list) > self.max_conv:
            for i in range(self.max_conv):
                self.json[self.uid_list[i]] = self.raw_json[self.uid_list[i]]
        else:
            self.json = self.raw_json

    def __len__(self):
        return min(len(self.uid_list), self.max_conv) * self.split

    def __getitem__(self, idx):
        uid = self.uid_list[idx // self.split]
        nbest_id = 'hyp_' + str(idx % self.split + 1)
        text = self.json[uid][nbest_id]['text']
        return {'text': text, 'conv_uid': uid, 'nbest_id': nbest_id}


class MLMScorer():

    def __init__(self):
        super().__init__()
        self.loader = None
        self.model = None
        self.load_model('roberta.base')

    # load_model: load model, and download it if it doesn't exist. For now, it is assumed that the model is roberta.
    def load_model(self, model_name):
        full_model_name = 'models/' + model_name + '.pt'
        if not os.path.exists(full_model_name):
            print(f"{model_name} model not found on models/ directory. Downloading from torch.hub ....")
            pretrained = torch.hub.load('pytorch/fairseq', model_name)
            torch.save(pretrained.model, full_model_name)

        pretrained = RobertaModel.from_pretrained(model_name)
        pretrained.eval()
        self.model = pretrained

    def calculate_wer(self, json_dataset):
        data = copy.deepcopy(json_dataset)
        total_wer = 0
        for conv_uid in data:
            best_sentence, reference = self.get_best_sentence_and_ref(data[conv_uid])
            total_wer += wer(reference, best_sentence)
        wer_percent = (total_wer / len(data)) * 100
        print(f"On {len(data)} unique conversation(s)\nWER: {wer_percent} %")
        return wer_percent

    def get_best_sentence_and_ref(self, convs):
        reference = convs.pop('ref')
        best_sentence = [y['text'] for y in sorted(convs.values(), key=lambda x: x['score'], reverse=True)][0]
        return best_sentence, reference


    def _init_loader(self, dataset):
        self.loader = DataLoader(dataset=dataset, shuffle=False)
        
    # SLOW, since it inference sentences one by one, token by token.
    def score_sentence(self, text):
        if not self.model:
            raise Exception("You should initialize model first by self.load_model(MODEL_NAME)")
        sum_loss = 0
        tokens = self.model.encode(text[0])
        # According to the paper's appendix A.1 scoring, 
        # conditional probabilities for [CLS] and [SEP] are not included.
        for i in range(1, len(tokens) - 1):
            orig_idx = tokens[i].item()
            # replace each position to <mask>
            tokens[i] = 50264
            output = self.model.model(tokens.unsqueeze(0))
            softmax = F.log_softmax(output[0][0][0], dim=0)
            sum_loss += softmax[orig_idx]
            # reset <mask> to original token idx
            tokens[i] = orig_idx
        return sum_loss.item()

    """
    get_pppl_scored_dataset: main part of the paper. Illustrated in Figure 1.
        1. Mask one token in each generated hypothesis sentence, and do this for every token in sentence. 
        2. Do a forward pass for each masked sentence. (done in score_sentence function)
        3. Softmax the output, and get the log probability for the original masked out token, by softmax_output[original_token_index]
        4. Sum the values up for each sentence.
    """

    def get_pppl_scored_dataset(self, dataset):
        res_json = copy.deepcopy(dataset.json)
        self._init_loader(dataset)
        for i, data in enumerate(tqdm.tqdm(self.loader)):
            # mask every token in sentence one at a time, and calculate PPPL.
            pppl_score = self.score_sentence(data['text'])
            bef = res_json[data['conv_uid'][0]][data['nbest_id'][0]]['score']
            res_json[data['conv_uid'][0]][data['nbest_id'][0]]['score'] = pppl_score

        return res_json

    def save_json(self, dataset, save_path):
        with open(save_path, 'w') as f:
            f.write(json.dumps(dataset))

    """
    rescore: implemented sequence-to-sequence rescoring, which decomposes prev score and newly calculated score by certain weight.
            In the paper, it is described that the value lambda is found by grid search.
    """
    def rescore(self, orig_dataset, pppl_dataset, scale=0.5):
        rescored_dataset = copy.deepcopy(pppl_dataset)
        for conv_uid in rescored_dataset:
            for hyp_uid in rescored_dataset[conv_uid]:
                if hyp_uid == 'ref':
                    continue
                rescored_dataset[conv_uid][hyp_uid]['score'] = scale * orig_dataset[conv_uid][hyp_uid]['score'] + (1 - scale) * pppl_dataset[conv_uid][hyp_uid]['score']
        return rescored_dataset    


if __name__ == "__main__":
    main()
