from torch.utils.data import Dataset, DataLoader
import json
import torch.nn.functional as F
import copy
import argparse
import os
import torch
from fairseq.models.roberta import RobertaModel


def get_args():
    parser = argparse.ArgumentParser(description='Preprocess librispeech json type dataset.')
    parser.add_argument('--PATH', help='json file path')
    parser.add_argument('--SUBSET', help='name of subset([dev/test])')
    parser.add_argument('--GROUP', help='name of subset group([clean/other])')
    parser.add_argument('--SAVE_PATH', help='path to save the preprocessed file')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    dataset = JSONDataset(args)
    scorer = MLMScorer()
    scorer.run_scoring(dataset)
    print("Run done!")
    scorer.save_json('res.json')
    return

class JSONDataset(Dataset):
    
    def __init__(self, args):
        super().__init__()
        self.file_path = args.PATH
        self.subset = args.SUBSET
        self.group = args.GROUP
        self.path = args.SAVE_PATH
        self.json = None
        self.uid_list = None
        self.load(self.file_path)

    def load(self, file_path):
        with open(file_path) as f:
            self.json = json.load(f)
            self.uid_list = list(self.json.keys())

    def __len__(self):
        return len(self.uid_list) * 99

    def __getitem__(self, idx):
        uid = self.uid_list[idx // 99]
        nbest_id = 'hyp_' + str(idx % 99 + 1) # hyp_1 ~ hyp_99
        conv = self.json[uid]
        orig_score = conv[nbest_id]['score']
        text = conv[nbest_id]['text']
        return {'text': text, 'conv_uid': uid, 'nbest_id': nbest_id, 'ref': conv['ref'],
                'orig_score': orig_score, 'idx': idx}




class MLMScorer():

    def __init__(self):
        super().__init__()
        self.loader = None
        self.model = None
        self.load_model('roberta.base')

    # load_model: for now, it is assumed that the model is roberta.
    def load_model(self, model_name):
        full_model_name = model_name + '.pt'
        if not os.path.exists(full_model_name):
            pretrained = torch.hub.load('pytorch/fairseq', model_name)
            torch.save(pretrained.model, full_model_name)

        pretrained = RobertaModel.from_pretrained(model_name)
        pretrained.eval()
        self.model = pretrained


    def _init_loader(self, dataset):
        self.loader = DataLoader(dataset=dataset, shuffle=False)

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
        return sum_loss


    def run_scoring(self, dataset):
        res_json = copy.deepcopy(dataset.json)
        self._init_loader(dataset)
        for data in self.loader:
            # mask every token in sentence one at a time, and calculate PPPL.
            pppl_score = self.score_sentence(data['text'])
            bef = res_json[data['conv_uid'][0]][data['nbest_id'][0]]['score']
            res_json[data['conv_uid'][0]][data['nbest_id'][0]]['score'] = pppl_score
        return res_json

    def save_json(self, save_path):
        with open(save_path, 'w') as f:
            f.write(json.dumps(self.res_json))


if __name__ == "__main__":
    main()
