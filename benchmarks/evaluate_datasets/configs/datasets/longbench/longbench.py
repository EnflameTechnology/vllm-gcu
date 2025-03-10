from mmengine.config import read_base

with read_base():
    from .longbenchpassage_retrieval_en.longbench_passage_retrieval_en_gen import LongBench_passage_retrieval_en_datasets
    from .longbenchpassage_retrieval_zh.longbench_passage_retrieval_zh_gen import LongBench_passage_retrieval_zh_datasets
    from .longbenchpassage_count.longbench_passage_count_gen import LongBench_passage_count_datasets
    from .longbenchtrec.longbench_trec_gen import LongBench_trec_datasets
    from .longbenchlsht.longbench_lsht_gen import LongBench_lsht_datasets
longbench_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])