import json
import re
from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.datasets.base import BaseDataset
from opencompass.datasets import humaneval_postprocess_v2
from opencompass.utils import get_data_path


@LOAD_DATASET.register_module()
class HumanevalStarcoderDataset(BaseDataset):

    @staticmethod
    def load(path: str, num_repeats: int = 1, local_mode: bool = False):
        path = get_data_path(path, local_mode=local_mode)
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                if data['prompt'].endswith('\n'):
                    data['prompt'] = data['prompt'][:-1]
                dataset.extend([data for _ in range(num_repeats)])
        return Dataset.from_list(dataset)


def humaneval_starcoder2_postprocess(text: str) -> str:
    end_words = [
        "\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif", "\n```", "<file_sep>", "<|endoftext|>",
    ]
    min_stop_index = len(text)
    for w in end_words:
        if w in text:
            stop_index = text.find(w)
            if stop_index != -1 and stop_index < min_stop_index:
                min_stop_index = stop_index
                text = text[:min_stop_index]
    print(text)
    # text = humaneval_postprocess_v2(text)
    return text


@LOAD_DATASET.register_module()
class HumanevalWizardcoderDataset(BaseDataset):
    @staticmethod
    def load(path: str, num_repeats: int = 1, local_mode: bool = False):
        path = get_data_path(path, local_mode=local_mode)
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                data['prompt'] = data['prompt'].replace("    ", "\t")
                dataset.extend([data for _ in range(num_repeats)])
        return Dataset.from_list(dataset)


def humaneval_wizardcoder_postprocess(text: str) -> str:
    text = text.split("### Response:")[-1]
    text = text.replace("\t", "    ")
    text = text.replace("\r", "")
    if "```python" in text:
        def_line = text.index("```python")
        text = text[def_line:].strip()
        text = text.replace("```python", "")
        try:
            next_line = text.index("```")
            text = text[:next_line].strip()
        except:
            print(text)
            print("================\n")
    if '__name__ == "__main__"' in text:
        next_line = text.index('if __name__ == "__main__":')
        text = text[:next_line].strip()

    if "# Example usage" in text:
        next_line = text.index("# Example usage")
        text = text[:next_line].strip()

    return text

# opencompass 0.2.1
def humaneval_postprocess(text: str) -> str:
    if '```' in text:
        blocks = re.findall(r'```(.*?)```', text, re.DOTALL)
        if len(blocks) == 0:
            text = text.split('```')[1]  # fall back to default strategy
        else:
            text = blocks[0]  # fetch the first code block
            if not text.startswith('\n'):  # in case starting with ```python
                text = text[max(text.find('\n') + 1, 0):]
    if text.strip().startswith('from') or text.strip().startswith('import'):
        def_idx = text.find('def')
        if def_idx != -1:
            text = text[max(text.find('\n', def_idx) + 1, 0):]
    text = text.split('\n\n')[0]
    text = text.lstrip('\n')
    if text.strip().startswith('def'):
        text = '\n'.join(text.split('\n')[1:])
    if not text.startswith('    '):
        if text.startswith(' '):
            text = '    ' + text.lstrip()
        else:
            text = '\n'.join(['    ' + line for line in text.split('\n')])
    return text