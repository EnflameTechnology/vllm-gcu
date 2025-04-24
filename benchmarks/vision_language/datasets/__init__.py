import importlib

VLM_DATASETS = {
    "llava-bench-coco": ("llava_bench_coco", "LlavaBenchCocoDataset"),
    "MMMU": ("mmmu", "MMMUDataset"),
    "videomme":("videomme", "VideoMMEDataset"),
    "fox_benchmark_data":("fox_benchmark_data", "FoxBenchmarkData")
}


def resolve_dataset_cls(dataset_name: str):
    dataset, dataset_cls_name = VLM_DATASETS[dataset_name]
    module = importlib.import_module(
        f"vllm_utils.vision_language.datasets.{dataset}")
    return getattr(module, dataset_cls_name, None)
