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
    orig_dataset = JSONDataset(args.ORIG_PATH)
    pppl_dataset = JSONDataset(args.PPPL_PATH)
    scorer = MLMScorer()
    rescore_path = args.ORIG_PATH.split('.json')[0] + '.rescored.json'
    scorer.rescore(orig_dataset, pppl_dataset, rescore_path, scale=args.SCALE)
    print(f"Saved rescored dataset at {rescore_path}.")
    score = scorer.calculate_wer(dataset.json)


def main():
    args = get_args()
    if args.PPPL_SCORE:
        pppl_score(args)
    elif args.RESCORE:
        rescore(args)
    
class JSONDataset(Dataset):
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.json = None
        self.uid_list = None
        self.load_json()

    def load_json(self):
        with open(self.file_path) as f:
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
        print(f"On {len(data)} unique conversation(s)\nWER: {wer_percent}%")
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
            if i > 500:
                break
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
    def rescore(self, orig_dataset, pppl_dataset, save_path, scale=0.5):
        rescored_dataset = copy.deepcopy(pppl_dataset)
        for conv_uid in rescored_dataset:
            for hyp_uid in rescored_dataset[conv_uid]:
                if hyp_uid == 'ref':
                    continue
                rescored_dataset[conv_uid][hyp_uid]['score'] = scale * orig_dataset[conv_uid][hyp_uid]['score'] + (1 - scale) * pppl_dataset[conv_uid][hyp_uid]['score']
        self.calculate_wer(rescored_dataset)
        self.save_json(rescored_dataset, save_path)
        return rescored_dataset    


if __name__ == "__main__":
    main()
