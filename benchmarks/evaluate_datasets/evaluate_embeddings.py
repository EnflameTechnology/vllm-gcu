import argparse
import os
import dataclasses
import json
import time
import csv


from collections.abc import Sequence

from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm import LLM
import mteb
import torch
import random

CMTEB_TASK_LIST = [
    "TNews",
    "IFlyTek",
    "MultilingualSentiment",
    "JDReview",
    "OnlineShopping",
    "Waimai",
    "AmazonReviewsClassification",
    "MassiveIntentClassification",
    "MassiveScenarioClassification",
    "MultilingualSentiment",
    "CLSClusteringS2S",
    "CLSClusteringP2P",
    "ThuNewsClusteringS2S",
    "ThuNewsClusteringP2P",
    "Ocnli",
    "Cmnli",
    "T2Reranking",
    "MmarcoReranking",
    "CMedQAv1",
    "CMedQAv2",
    "T2Retrieval",
    "MMarcoRetrieval",
    "DuRetrieval",
    "CovidRetrieval",
    "CmedqaRetrieval",
    "EcomRetrieval",
    "MedicalRetrieval",
    "VideoRetrieval",
    "ATEC",
    "BQ",
    "LCQMC",
    "PAWSX",
    "STSB",
    "AFQMC",
    "QBQTC",
    "STS22",
]


class VLLM_MODEL:
    def __init__(self, model):
        self.model = model

    def encode(
        self,
        sentences: Sequence[str],
        *args,
        **kwargs,
    ):
        if "batch_size" in kwargs:
            batch_size = kwargs["batch_size"]
            outputs = []
            for i in range(0, len(sentences), batch_size):
                output = self.model.encode(sentences[i : i + batch_size])
                outputs.extend(output)
        else:
            outputs = self.model.encode(sentences)
        res = [output.outputs.data for output in outputs]
        stacked_tensor = torch.stack(res)
        return stacked_tensor.numpy()


def get_vllm_model(engine_args, **kwargs):
    llm_build_dict = dataclasses.asdict(engine_args)
    llm_build_dict.update(kwargs)
    llm = LLM(**llm_build_dict)
    return VLLM_MODEL(llm)


def main(args):
    print(args)
    random.seed(args.seed)
    msgs = vars(args)
    if args.backend == "vllm":
        model = get_vllm_model(EngineArgs.from_cli_args(args))
    else:
        raise ValueError("Unsupported backend!")
    task = args.eval_task
    eval_split = args.eval_split
    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    if not os.path.exists(dataset_dir):
        raise ValueError(f"Can not find {dataset_dir} from local file system")
    if eval_split:
        tasks = mteb.get_tasks(tasks=[task], eval_splits=[eval_split])
    else:
        tasks = mteb.get_tasks(tasks=[task])
    evaluation = mteb.MTEB(tasks=tasks)
    evaluation.tasks[0].metadata.dataset["path"] = dataset_dir
    if batch_size:
        encode_kwargs = {"batch_size": batch_size}
    else:
        encode_kwargs = {}
    res_list = evaluation.run(
        model, encode_kwargs=encode_kwargs, overwrite_results=True
    )
    for res in res_list:
        print(res)
    msgs.update(res_list[0].scores)
    csvfile = save_to_csv(f"{time.strftime('%Y%m%d%H%M%S')}.csv", msgs)
    print(f"save to {csvfile}")
    if args.save_output:
        with open(args.save_output, "w", encoding="UTF-8") as f:
            json.dump(res_list[0].scores, f, indent=4)


def save_to_csv(csv_file, data):

    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Key", "Value"])
        writer.writerows(data.items())

    return csv_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark embedding model by mteb"
    )
    parser.add_argument("--backend", type=str, choices=["vllm"], default="vllm")

    parser.add_argument(
        "--dataset-dir", type=str, default=None, help="dir to the dataset."
    )
    parser.add_argument(
        "--eval-task",
        type=str,
        default=None,
        help=f"name for mteb task such as {CMTEB_TASK_LIST}",
    )
    parser.add_argument(
        "--eval-split", type=str, default=None, help="split for mteb task"
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-output", type=str, default=None)
    parser = AsyncEngineArgs.add_cli_args(parser)

    args = parser.parse_args()
    if args.tokenizer is None:
        args.tokenizer = args.model
    main(args)
