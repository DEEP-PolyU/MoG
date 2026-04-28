
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import yaml

from models.utils.logger import logger


@dataclass
class DatasetConfig:
    corpus_path: str
    qa_path: str
    schema_path: str
    schema_path_base: str
    graph_output: str

@dataclass
class TriggersConfig:
    constructor_trigger: bool = True
    retrieve_trigger: bool = True
    mode: str = "agent"

@dataclass
class ConstructionConfig:
    max_workers: int = 32
    datasets_no_chunk: list = None
    chunk_size: int = 1000
    overlap: int = 200
    memory_optimization: Optional['MemoryOptimizationConfig'] = None
    use_batch_processing: bool = False
    batch_size: int = 100

    def __post_init__(self):
        if self.datasets_no_chunk is None:
            self.datasets_no_chunk = ["hotpot", "2wiki", "musique"]

@dataclass
class TreeCommConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    struct_weight: float = 0.3
    enable_fast_mode: bool = True
    max_total_subGraphs: int = 100

@dataclass
class SubGraphDetectionConfig:
    min_subGraph_size: int = 10
    n_clusters: Optional[int] = None
    enable_soft_clustering: bool = False
    connectivity_weight: float = 0.3
    cohesion_weight: float = 0.2
    overlap_weight: float = 0.5

@dataclass
class MemoryStrategiesConfig:
    use_sparse_matrices: bool = True
    enable_gc: bool = True
    processing_strategy: str = "hierarchical"
    disable_subgraph_construction: bool = True
    log_memory_usage: bool = True
    fallback_enabled: bool = True
    use_degree_fallback: bool = True

@dataclass
class MemoryOptimizationConfig:
    enabled: bool = True
    max_nodes_threshold: int = 20000
    sampling_ratio: float = 0.2
    max_nodes_per_batch: int = 5000
    subGraph_detection: SubGraphDetectionConfig = None
    strategies: MemoryStrategiesConfig = None

    def __post_init__(self):
        if self.subGraph_detection is None:
            self.subGraph_detection = SubGraphDetectionConfig()
        if self.strategies is None:
            self.strategies = MemoryStrategiesConfig()

@dataclass
class FAISSConfig:
    max_workers: int = 4
    device: str = "cpu"

@dataclass
class RetrievalConfig:
    top_k: int = 5
    recall_paths: int = 2
    similarity_threshold: float = 0.3
    enable_caching: bool = True
    faiss: FAISSConfig = FAISSConfig()

@dataclass
class EmbeddingsConfig:
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    batch_size: int = 32
    max_length: int = 512

@dataclass
class NLPConfig:
    spacy_model: str = 'en_core_web_lg' 


@dataclass
class OutputConfig:
    base_dir: str = "output"
    project_dir: str = "deepseek-v3"
    graphs_dir: str = "output/graphs_mog"
    graphs_processing_dir: str = "output/graphs_preprocessing"
    chunks_dir: str = "output/chunks"
    logs_dir: str = "output/logs"
    cache_dir: str = "output/faiss_cache"
    hub_path_name: str = "hub"
    graphs_meta_data_dir: str = "output/experiment_results"
    results_dir: str = "results"
    checkpoints_dir: str = "output/checkpoints"

@dataclass
class MoGConfig:
    min_subGraph_size: int = 20
    weight_semantic: float = 0.7
    weight_bridging: float = 0.3
    min_activated_experts: int = 5
    max_activated_experts: int = 10
    min_expert_size: int = 20
    min_hub_degree: int = 0
    hub_percentile: int = 96

@dataclass
class PerformanceConfig:
    parallel_processing: bool = True
    max_workers: int = 32
    batch_size: int = 16
    memory_optimization: bool = True

@dataclass
class EvaluationConfig:
    enable_evaluation: bool = True
    metrics: list = None
    save_detailed_results: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "precision", "recall", "f1"]

class ConfigManager:
    
    def __init__(self, config_path: Optional[str] = None, construction_mode="", retrieval_mode=""):
        self.config_path = config_path or self._get_default_config_path()
        self.config_data: Dict[str, Any] = {}
        self.datasets: Dict[str, DatasetConfig] = {}
        self.triggers: Optional[TriggersConfig] = None
        self.construction: Optional[ConstructionConfig] = None
        self.retrieval: Optional[RetrievalConfig] = None
        self.embeddings: Optional[EmbeddingsConfig] = None
        self.nlp: Optional[NLPConfig] = None
        self.prompts: Dict[str, Any] = {}
        self.output: Optional[OutputConfig] = None
        self.mixture_of_graph: Optional[MoGConfig] = None
        self.performance: Optional[PerformanceConfig] = None
        self.evaluation: Optional[EvaluationConfig] = None
        self.memory_optimization: Optional[MemoryOptimizationConfig] = None

        self.construction_mode=construction_mode
        self.retrieval_mode=retrieval_mode

        self.load_config()
    
    def _get_default_config_path(self) -> str:
        current_dir = Path(__file__).parent
        return str(current_dir / "base_config.yaml")
    
    def load_config(self) -> None:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)

            hub_percentile = self.config_data['mixture_of_graph']['hub_percentile']
            hub_path_name= f"hub_{hub_percentile}"

            base_dir = self.config_data['output']['base_dir']
            project_dir =  self.config_data['output']['project_dir']
            self.config_data['output']['cache_dir'] = os.path.join(base_dir, project_dir, "faiss_cache", self.construction_mode)
            self.config_data['output']['hub_path_name'] = hub_path_name

            self.config_data['output']['graphs_dir'] = os.path.join(base_dir, project_dir, "graphs_mog", self.construction_mode, hub_path_name)
            self.config_data['output']['graphs_meta_data_dir'] = os.path.join(base_dir, project_dir, "graphs_mog", "graphs_mog_metadata", self.construction_mode, hub_path_name)
            self.config_data['output']['results_dir'] = os.path.join("results", project_dir, self.construction_mode, hub_path_name)

            self.config_data['output']['checkpoints_dir'] = os.path.join(base_dir, project_dir, "checkpoints")
            self.config_data['output']['graphs_processing_dir'] = os.path.join(base_dir, project_dir,"graphs_preprocessing")
            self.config_data['output']['chunks_dir'] = os.path.join(base_dir, project_dir, "chunks")
            self.config_data['output']['logs_dir'] = os.path.join(base_dir, project_dir, "logs")

            output_paths = [
                self.config_data['output']['graphs_dir'],
                self.config_data['output']['cache_dir'],
                self.config_data['output']['graphs_meta_data_dir'],
                self.config_data['output']['results_dir'],
                self.config_data['output']['checkpoints_dir'],
                self.config_data['output']['graphs_processing_dir'],
                self.config_data['output']['chunks_dir'],
                self.config_data['output']['logs_dir']
            ]

            for path in output_paths:
                os.makedirs(path, exist_ok=True)

            self._parse_config()
            self._validate_config()
            
            logger.info(f"Configuration loaded successfully from {self.config_path}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in configuration file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def _parse_config(self) -> None:
        datasets_data = self.config_data.get("datasets", {})
        self.datasets = {
            name: DatasetConfig(**config) 
            for name, config in datasets_data.items()
        }
        
        triggers_data = self.config_data.get("triggers", {})
        self.triggers = TriggersConfig(**triggers_data)
        
        construction_data = self.config_data.get("construction", {})
        memory_opt_data = construction_data.pop("memory_optimization", {})

        self.construction = ConstructionConfig(**construction_data)

        if memory_opt_data:
            subGraph_detection_data = memory_opt_data.pop("subGraph_detection", {})
            strategies_data = memory_opt_data.pop("strategies", {})

            memory_opt_config = MemoryOptimizationConfig(**memory_opt_data)
            if subGraph_detection_data:
                memory_opt_config.subGraph_detection = SubGraphDetectionConfig(**subGraph_detection_data)
            if strategies_data:
                memory_opt_config.strategies = MemoryStrategiesConfig(**strategies_data)

            self.construction.memory_optimization = memory_opt_config

        retrieval_data = self.config_data.get("retrieval", {})
        faiss_data = retrieval_data.get("faiss", {})
        self.retrieval = RetrievalConfig(**retrieval_data)
        self.retrieval.faiss = FAISSConfig(**faiss_data)

        logger.warning(f"retrieval_data {retrieval_data}")
        logger.warning(f"faiss_data {faiss_data}")
        logger.warning(f"self.retrieval {self.retrieval}")
        logger.warning(f"self.retrieval.faiss {self.retrieval.faiss}")
        
        embeddings_data = self.config_data.get("embeddings", {})
        self.embeddings = EmbeddingsConfig(**embeddings_data)
        
        nlp = self.config_data.get("nlp", {})
        self.nlp = NLPConfig(**nlp)
        
        self.prompts = self.config_data.get("prompts", {})
        
        output_data = self.config_data.get("output", {})
        self.output = OutputConfig(**output_data)

        mog_data = self.config_data.get("mixture_of_graph", {})
        self.mixture_of_graph= MoGConfig(**mog_data)

        
        performance_data = self.config_data.get("performance", {})
        self.performance = PerformanceConfig(**performance_data)
        
        evaluation_data = self.config_data.get("evaluation", {})
        self.evaluation = EvaluationConfig(**evaluation_data)
    
    def _validate_config(self) -> None:
        for dataset_name, dataset_config in self.datasets.items():
            if not os.path.exists(dataset_config.corpus_path):
                logger.warning(f"Corpus path not found for {dataset_name}: {dataset_config.corpus_path}")
            if not os.path.exists(dataset_config.schema_path_base):
                logger.warning(f"Schema path not found for {dataset_name}: {dataset_config.schema_path_base}")
            if not os.path.exists(dataset_config.schema_path):
                logger.warning(f"Schema path not found for {dataset_name}: {dataset_config.schema_path}")

        if self.retrieval.top_k <= 0:
            raise ValueError("top_k must be positive")
    
    def get_dataset_config(self, dataset_name: str, retrieval_trigger=False) -> DatasetConfig:
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found in configuration")

        dataset_config = DatasetConfig(
            corpus_path=self.datasets[dataset_name].corpus_path,
            qa_path=self.datasets[dataset_name].qa_path,
            schema_path_base=self.datasets[dataset_name].schema_path_base,
            schema_path=self.datasets[dataset_name].schema_path,
            graph_output=os.path.join(self.config_data['output']['graphs_dir'], self.datasets[dataset_name].graph_output)
        )

        return dataset_config

    def get_prompt(self, category: str, prompt_type: str) -> str:
        try:
            return self.prompts[category][prompt_type]
        except KeyError:
            raise ValueError(f"Prompt not found: {category}.{prompt_type}")
    
    def get_prompt_formatted(self, category: str, prompt_type: str, **kwargs) -> str:
        template = self.get_prompt(category, prompt_type)
        try:
            return template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable {e} for prompt {category}.{prompt_type}")
    
    def override_config(self, overrides: Dict[str, Any]) -> None:
        def update_nested_dict(d: dict, overrides: dict) -> None:
            for key, value in overrides.items():
                if isinstance(value, dict) and key in d and isinstance(d[key], dict):
                    update_nested_dict(d[key], value)
                else:
                    d[key] = value
        
        update_nested_dict(self.config_data, overrides)
        self._parse_config()
        self._validate_config()
    
    def save_config(self, output_path: str) -> None:
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config_data, f, default_flow_style=False, ensure_ascii=False)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "datasets": {name: asdict(config) for name, config in self.datasets.items()},
            "triggers": asdict(self.triggers),
            "construction": asdict(self.construction),
            "retrieval": asdict(self.retrieval),
            "embeddings": asdict(self.embeddings),
            "prompts": self.prompts,
            "output": asdict(self.output),
            "mixture_of_graph": asdict(self.mixture_of_graph),
            "performance": asdict(self.performance),
            "evaluation": asdict(self.evaluation),
        }


    def create_output_directories(self) -> None:
        directories = [
            self.output.base_dir,
            self.output.graphs_dir,
            self.output.chunks_dir,
            self.output.logs_dir,
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

_config_instance: Optional[ConfigManager] = None

def get_config(config_path: Optional[str] = None, construction_mode="MoGConstruction", retrieval_mode="") -> ConfigManager:
    global _config_instance
    
    if _config_instance is None:
        _config_instance = ConfigManager(config_path, construction_mode, retrieval_mode)
    
    return _config_instance

def reload_config(config_path: Optional[str] = None) -> ConfigManager:
    global _config_instance
    _config_instance = ConfigManager(config_path)
    return _config_instance
