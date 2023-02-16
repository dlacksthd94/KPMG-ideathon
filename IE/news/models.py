import pororo
from tqdm import tqdm
import os
import platform
import zipfile
import wget

import torch
from fairseq.models.roberta import RobertaHubInterface
from fairseq import hub_utils
# from pororo.tasks.utils.config import CONFIGS

class MyFastPororo():
    def __init__(self):
        # tag full name
        self._tag = {
            "PS": "PERSON",
            "LC": "LOCATION",
            "OG": "ORGANIZATION",
            "AF": "ARTIFACT",
            "DT": "DATE",
            "TI": "TIME",
            "CV": "CIVILIZATION",
            "AM": "ANIMAL",
            "PT": "PLANT",
            "QT": "QUANTITY",
            "FD": "STUDY_FIELD",
            "TR": "THEORY",
            "EV": "EVENT",
            "MT": "MATERIAL",
            "TM": "TERM",
        }
        
        # config
        self.task='ner'
        self.lang = "ko"
        self.model_name = "charbert.base.ko.ner"
        self.apply_wsd: bool = False
        self.ignore_labels: list = []
        self.add_special_tokens: bool = True
        self.no_separator: bool = False
        self.addl_sentences = {}
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_dir = None
        self.model = None
        
        self.load_model()

    def download_model(self):
        url = 'https://twg.kakaocdn.net/pororo/ko/models/bert/charbert.base.ko.ner.zip'

        pf = platform.system()

        if pf == "Windows":
            save_dir = "C:\\pororo"
        else:
            home_dir = os.path.expanduser("~")
            save_dir = os.path.join(home_dir, ".pororo")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        root_dir = save_dir # ~/.pororo
        type_dir = os.path.join(root_dir, 'bert')
        model_path = os.path.join(type_dir, self.model_name)
        zip_path = model_path + '.zip'

        if not os.path.exists(os.path.join(type_dir, self.model_name)):
            os.makedirs(type_dir, exist_ok=True)
            wget.download(url, type_dir)
            zip_file = zipfile.ZipFile(zip_path)
            zip_file.extractall(type_dir)
            zip_file.close()
        
        self.ckpt_dir = model_path
    
    def load_model(self):
        self.download_model()
        
        kwargs = {}
        x = hub_utils.from_pretrained(
            self.ckpt_dir,
            "model.pt",
            self.ckpt_dir,
            **kwargs,
        )
        
        self.model = RobertaHubInterface(
            x["args"],
            x["task"],
            x["models"][0],
        )

    def tokenizer(self, x):
        """ 
        input (str): a sentence to tokenize
        output (str, tensor): a tokenized sentence and its token_ids
        """
        x = x.strip()
        if len(x) == 0:
            result = ""
        else:
            # x = [c for c in re.sub("\s+", " ", x)]
            x = list(x)

            result = list()
            for i in range(len(x)):
                if x[i] == " ":
                    x[i + 1] = f"▁{x[i+1]}"
                    continue
                else:
                    result.append(x[i])
            result[0] = f"▁{result[0]}"
            result = " ".join(result)
            bpe_sentence = result

        if self.add_special_tokens:
            bpe_sentence = f"<s> {result} </s>"

        # tokens in number code
        tokens = self.model.task.source_dictionary.encode_line(
            bpe_sentence,
            append_eos=False,
            add_if_not_exist=False,
        )
        tokens = tokens.long()
        return result, tokens

    def inference(self, inputs):
        # inference into class code
        preds = (self.model.predict(
            "sequence_tagging_head",
            inputs,
        )[:, 1:-1, :].argmax(dim=2).cpu().numpy())

        return preds

    def post_process(self, tokenized_sent, preds):
        # class code into label
        label_fn = lambda label: self.model.task.label_dictionary.string([label])
        labels = [
            label_fn(int(pred) + self.model.task.label_dictionary.nspecial)
            for pred in preds
        ]

        # make pairs of (token, label)
        res = [
            (token, label)
            for token, label in zip(tokenized_sent.split(), labels)
        ]

        def _remove_tail(tag: str):
            if "-" in tag:
                tag = tag[:-2]
            return tag

        def _postprocess(tags):
            result = list()

            tmp_word = tags[0][0]
            prev_ori_tag = tags[0][1]
            prev_tag = _remove_tail(prev_ori_tag)
            for _, pair in enumerate(tags[1:]):
                char = pair[0]
                ori_tag = pair[1]
                tag = _remove_tail(ori_tag)
                if ("▁" in char) and ("-I" not in ori_tag):
                    result.append((tmp_word, prev_tag))
                    result.append((" ", "O"))

                    tmp_word = char
                    prev_tag = tag
                    continue

                if (tag == prev_tag) and (("-I" in ori_tag) or "O" in ori_tag):
                    tmp_word += char
                elif (tag != prev_tag) and ("-I" in ori_tag) and (tag != "O"):
                    tmp_word += char
                else:
                    result.append((tmp_word, prev_tag))
                    tmp_word = char

                prev_tag = tag
            result.append((tmp_word, prev_tag))

            result = [(pair[0].replace("▁", " ").strip(),
                    pair[1]) if pair[0] != " " else (" ", "O")
                    for pair in result]
            return result

        # concat input into words
        res = [
            pair for pair in _postprocess(res)
            if pair[1] not in self.ignore_labels
        ]

        # abb fag into full tag
        res = [(
            pair[0],
            self._tag[pair[1]],
        ) if pair[1] in self._tag else pair for pair in res]

        # res = res if not self.apply_wsd else self._apply_wsd(res)
        return res

def main():
    # prepare model
    ner = MyFastPororo()
    ner.load_model()

    # inference
    text = '삼정 KPMG에서 주최하는 아이디어톤에 서울대 데이터사이언스대학원 (GSDS) 의 박건도, 서아름, 손성욱, 임찬송, 최유림, 허상우 학생이 참여하였습니다.'

    tokenized_sent, token_ids = ner.tokenizer(text)
    preds = ner.inference(token_ids)
    result = ner.post_process(tokenized_sent, preds)
    
    print(result)
    
if __name__== "__main__":
    main()