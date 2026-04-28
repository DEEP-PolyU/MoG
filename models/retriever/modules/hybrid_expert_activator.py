
import numpy as np
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict
import torch


from ...utils import logger, LLMCompletionCall


class HybridExpertActivator:
    def __init__(
        self,
        config,
        expert_subGraphs: Dict[str, Any],
        entity_embedding_manager,
        chunk_embedding_manager,
        chunk_id2text,
        qa_encoder,
        full_graph,
        device: str = "cpu",
        max_activated_experts: int = 5,
    ):
        self.expert_subGraphs = expert_subGraphs
        self.entity_embedding_manager = entity_embedding_manager
        self.chunk_embedding_manager = chunk_embedding_manager
        self.qa_encoder = qa_encoder
        self.full_graph = full_graph
        self.device = device
        self.max_activated_experts = max_activated_experts
        self.chunk_id2text = chunk_id2text
        self.llm_client = LLMCompletionCall(config.output.results_dir)

        logger.info(f"🎯 Initialized HybridExpertActivator")

        if expert_subGraphs:
            sample_id = list(expert_subGraphs.keys())[0]
            sample_comm = expert_subGraphs[sample_id]
            logger.info(f"🔍 Sample expert subGraph '{sample_id}':")
            logger.info(f"   - Type: {type(sample_comm)}")
            logger.info(f"   - Has keywords attr: {hasattr(sample_comm, 'keywords')}")
            logger.info(f"   - Has description attr: {hasattr(sample_comm, 'description')}")
            if hasattr(sample_comm, 'keywords'):
                logger.info(f"   - Keywords value: {sample_comm.keywords}")
            if hasattr(sample_comm, 'description'):
                logger.info(f"   - Description value: {sample_comm.description[:100] if sample_comm.description else 'EMPTY'}")

        self.expert_subGraphs_content =  "=== Expert subGraphs ==="
        for expert_ids, subGraph_info in self.expert_subGraphs.items():
            keywords = subGraph_info.keywords if hasattr(subGraph_info, 'keywords') else []
            description = subGraph_info.description if hasattr(subGraph_info, 'description') else ''
            self.expert_subGraphs_content += f"\nExpert ID: {expert_ids}\n"
            self.expert_subGraphs_content += f"Keywords: {keywords}\n"
            self.expert_subGraphs_content += f"Description: {description[:400]}\n"


        self.entity_to_expert = self._build_entity_expert_mapping()


    def _build_entity_expert_mapping(self) -> Dict[str, str]:
        mapping = {}
        for expert_id, subGraph_info in self.expert_subGraphs.items():
            for entity_id in subGraph_info.members:
                if entity_id in mapping:
                    mapping[entity_id].append(expert_id)
                else:
                    mapping[entity_id] = [expert_id]
        return mapping


    def activate_experts_by_hubs(
            self,
            topk_shared_entity_ids: List[str],
            retrieval_mode:str
    ) -> Tuple[List[str], Dict[str, Any]]:
        if self.max_activated_experts == 0:
            activated = []
            activation_info = {
                'semantic_scores': {},
                'bridging_scores': {},
                'chunk_scores': {},
                'final_scores': {},
                'total_candidates': 0,
                'activated_count': 0
            }
            return activated, activation_info
        logger.info(f"🎯 Activating experts via hybrid strategy")

        bridging_scores = self._compute_bridging_scores(topk_shared_entity_ids)
        bridging_scores = self._normalize_scores(bridging_scores)
        all_experts = set(bridging_scores.keys())
        final_scores = {}

        for expert_id in all_experts:
            bri_score = bridging_scores.get(expert_id, 0.0)
            score = bri_score
            if score > 0:
                final_scores[expert_id] = score
        sorted_experts = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        if "hubActBonlyFull" in retrieval_mode:
            activated = [expert_id for expert_id, _ in sorted_experts]
        else:
            activated = [expert_id for expert_id, _ in sorted_experts[:self.max_activated_experts]]
        logger.warning(f"✅ Activated {len(activated)} experts")
        for i, expert_id in enumerate(activated):
            logger.warning(f"   {i + 1}. {expert_id}: bridging={bridging_scores.get(expert_id, 0):.3f}")

        activation_info = {
            'bridging_scores': bridging_scores,
            'final_scores': {exp_id: final_scores[exp_id] for exp_id in activated},
            'total_candidates': len(final_scores.keys()),
            'activated_count': len(activated)
        }


        return activated, activation_info

    def _compute_bridging_scores(
            self,
            shared_entity_ids: List[str]
    ) -> Dict[str, float]:
        expert_touch_counts = defaultdict(float)
        for entity_id in shared_entity_ids:
            expert_id = self.entity_to_expert.get(entity_id, None)
            if expert_id is not None:
                for eid in expert_id:
                    expert_touch_counts[eid] += 1
        logger.warning(f"Computed bridging scores for {len(expert_touch_counts)} experts")
        return dict(expert_touch_counts)

    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        if not scores:
            return {}
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)

        if max_val - min_val < 1e-8:
            return {k: 1.0 for k in scores.keys()}

        normalized = {
            k: (v - min_val) / (max_val - min_val)
            for k, v in scores.items()
        }

        return normalized

