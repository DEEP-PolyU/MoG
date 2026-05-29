
import json
import json_repair
import time
import argparse
import os
import glob
import shutil

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

from models.constructor import mixture_of_graph_construction
from models.retriever.modules import GraphAgenticDecomposer
from models.retriever.mixture_of_graph_retriever import MixtureOfGraphRetriever
from models.retriever import mixture_of_graph_question_answer_pipeline


from config import get_config, ConfigManager
from models.utils.logger import logger


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/base_config.yaml"
    )
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        default=["demo"]
    )

    parser.add_argument(
        "--override",
        type=str
    )


    parser.add_argument(
        "--construction_mode",
        default="MoGBuild"
    )

    parser.add_argument(
        "--retrieval_mode",
        default="MoGRetrieval"
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None
    )

    return parser.parse_args()


def setup_environment(config: ConfigManager):
    config.create_output_directories()
    
    logger.info(f"Constructor enabled: {config.triggers.constructor_trigger}")
    logger.info(f"Retriever enabled: {config.triggers.retrieve_trigger}")


def clear_faiss_cache(dataset_name: str, config, clear_entityAndChunks_embedding=False) -> None:
    if clear_entityAndChunks_embedding:
        faiss_cache_dir = f"{config.output.cache_dir}/{dataset_name}"
    else:
        faiss_cache_dir = f"{config.output.cache_dir}/{dataset_name}/{config.output.hub_path_name}"

    if not os.path.exists(faiss_cache_dir):
        os.makedirs(faiss_cache_dir, exist_ok=True)
        logger.info(f"FAISS cache directory did not exist, created: {faiss_cache_dir}")
        return

    shutil.rmtree(faiss_cache_dir)
    logger.info(f"Cleared FAISS cache directory: {faiss_cache_dir}")
    os.makedirs(faiss_cache_dir, exist_ok=True)



def graph_construction(config, datasets, construction_mode="MoGBuild", retrieval_trigger=False):
    if config.triggers.constructor_trigger:
        logger.info(f"Starting knowledge graph construction in {construction_mode} mode...")

        for dataset in datasets:
            dataset_config = config.get_dataset_config(dataset, retrieval_trigger)
            logger.info(f"Building knowledge graph for dataset: {dataset}")

            logger.info("Graph construction mode: clearing FAISS and graph caches...")


            logger.info("Builder for knowledge graph construction initialized")
            builder = mixture_of_graph_construction.MixtureOfGraphConstructor(
                dataset,
                config=config,
                construction_mode=construction_mode,
                use_chunks_cache=True
            )
            if "MoGBuild" in construction_mode:
                logger.warning(f"Running subGraph detection experiment with Mixture-of-Graph")
                builder.build_preprocessed_knowledge_graph(dataset_config.corpus_path)
                builder.mog_subGraph_detection("Mixture-of-Graph")
                clear_faiss_cache(dataset, config, clear_entityAndChunks_embedding = False)
            elif "KGPreprocess" in construction_mode:
                logger.warning("Preprocessing: Building and saving knowledge graph...")
                builder.build_preprocessed_knowledge_graph(dataset_config.corpus_path)
                clear_faiss_cache(dataset, config, clear_entityAndChunks_embedding= True)
            else:
                logger.info("Error experiment-mode...")

            logger.info(f"Successfully built knowledge graph for {dataset}")
    return


def retrieval(config, datasets, resume_from=None, sample=None, retrieval_mode="rP-ETC_scoreAct_iterSubQ_rMOG"):
    for dataset_name in datasets:
        dataset_config = config.get_dataset_config(dataset_name, retrieval_trigger=True)

        logger.info(f"📂 Graph path from config: {dataset_config.graph_output}")

        with open(dataset_config.qa_path, "r") as f:
            qa_pairs = json_repair.load(f)

        if sample and sample < len(qa_pairs):
            logger.info(f"📊 Sampling {sample} questions from {len(qa_pairs)} total")
            qa_pairs = qa_pairs[:sample]

        graph_question_decomposer = GraphAgenticDecomposer(dataset_name, config=config)
        
        logger.info("🚀 Initializing retriever 🚀")
        logger.info("-"*30)

        logger.info("Using MixtureOfGraphRetrieverV2 for shared-expert subGraph detection")
        start_time = time.time()
        MoG_retriever = MixtureOfGraphRetriever(
            dataset_name=dataset_name,
            json_path=dataset_config.graph_output,
            cache_dir=config.output.cache_dir,
            chunks_dir=config.output.chunks_dir,
            schema_path=dataset_config.schema_path,
            top_k=config.retrieval.top_k,
            config=config,
            retrieval_mode=retrieval_mode
        )
        logger.info(f"Time taken to build MoG_retriever: {time.time() - start_time} seconds")
        logger.info("-"*30)

        logger.info(f"Start answering questions...")
        logger.info("-"*30)

        logger.info("🎯 Using Mixture of Graph V2 Retrieval System")
        if resume_from:
            logger.info(f"🔄 Resume mode enabled: {resume_from}")
        MoG_pipeline = mixture_of_graph_question_answer_pipeline.MixtureOfGraphQuestionAnswerPipeline(config=config)

        MoG_pipeline.mixture_of_graph_question_answer(config, graph_question_decomposer, MoG_retriever, qa_pairs,
            dataset_config.schema_path, resume_from=resume_from, retrieval_mode=retrieval_mode)



if __name__ == "__main__":
    args = parse_arguments()
    config_path = args.config
    config = get_config(config_path, args.construction_mode, args.retrieval_mode)
    
    if args.override:
        try:
            overrides = json.loads(args.override)
            config.override_config(overrides)
            logger.info("Applied configuration overrides")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in override parameter: {e}")
            exit(1)
    
    setup_environment(config)
    # ########### Construction ###########
    if config.triggers.constructor_trigger:
        logger.info("Starting knowledge graph construction...")
        graph_construction(
            config,
            args.datasets,
            construction_mode=args.construction_mode
        )

    # ########### Retriever ###########
    if config.triggers.retrieve_trigger:
        retrieval(config,args.datasets, resume_from=args.resume, sample=args.sample, retrieval_mode=args.retrieval_mode)
