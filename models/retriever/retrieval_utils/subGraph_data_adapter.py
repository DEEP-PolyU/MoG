
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict


from ...utils import logger

@dataclass
class SubGraphInfo:
    subGraph_id: str
    type: str
    member_count: int
    members: List[str]
    keywords: List[str]
    description: str
    node_id: str

    auto_discovered: bool = False

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            'subGraph_id': self.subGraph_id,
            'type': self.type,
            'member_count': self.member_count,
            'members': self.members,
            'keywords': self.keywords,
            'description': self.description,
            'node_id': self.node_id,
            'metadata': self.metadata
        }

        return data

    def __repr__(self) -> str:
        return f"SubGraphInfo(id={self.subGraph_id}, type={self.type}, members={self.member_count})"


class SubGraphDataAdapter:

    def __init__(self, mog_data: Dict, subGraphs: Dict):
        self.mog_data = mog_data
        self.subGraphs_dict = subGraphs
        self.subGraph_details_map = {}
        self.shared_subGraphs = {}
        self.expert_subGraphs = {}
        self.node_memberships_map = {}

        self._build_indices()

        logger.info(f"✅ SubGraphDataAdapter initialized: "
                   f"{len(self.shared_subGraphs)} shared, "
                   f"{len(self.expert_subGraphs)} expert subGraphs")

    def _build_indices(self):
        logger.info("Building subGraph data indices...")

        for comm_node_id, comm_node_data in self.subGraphs_dict.items():
            self.subGraph_details_map[comm_node_id] = comm_node_data
        logger.info(f"✅ Built details map for {len(self.subGraph_details_map)} subGraphs")
        sample_comm_ids = list(self.subGraph_details_map.keys())[:2]
        for comm_id in sample_comm_ids:
            detail = self.subGraph_details_map[comm_id]
            logger.info(f"🔍 Sample subGraph {comm_id}: "
                        f"keywords={len(detail.get('keywords', []))} items, "
                        f"description={len(detail.get('description', ''))} chars")

        self.node_memberships_map = self.mog_data.get('node_memberships', {})

        logger.debug(f"Total nodes with memberships: {len(self.node_memberships_map)}")

        shared_data = {}
        expert_data = {}
        for key, value in self.subGraphs_dict.items():
            if "Shared_" in key:
                shared_data[key] = value
            else:
                expert_data[key] = value

        shared_comm_ids = list(shared_data.keys())
        for comm_id in shared_comm_ids:
            detail = self.subGraph_details_map.get(comm_id, {})
            members = shared_data[comm_id]['member_nodes']

            logger.debug(f"Loading shared subGraph {comm_id}: {len(members)} members")

            self.shared_subGraphs[comm_id] = SubGraphInfo(
                subGraph_id=comm_id,
                type='shared_subGraph',
                member_count=len(members),
                members=members,
                keywords=detail.get('keywords'),
                description=detail.get('description'),
                node_id=detail.get('node_id'),
                metadata=detail
            )

        expert_comm_ids = list(expert_data.keys())
        for comm_id in expert_comm_ids:
            detail = self.subGraph_details_map.get(comm_id, {})

            members = expert_data[comm_id]['member_nodes']

            keywords = detail.get('keywords')
            description = detail.get('description')

            logger.debug(f"Loading expert subGraph {comm_id}:")
            logger.debug(f"  - Members: {len(members)}")
            logger.debug(f"  - Keywords: {keywords if keywords else 'EMPTY'}")
            logger.debug(f"  - Description: {description[:100] if description else 'EMPTY'}")
            logger.debug(f"  - Detail keys: {list(detail.keys())}")

            self.expert_subGraphs[comm_id] = SubGraphInfo(
                subGraph_id=comm_id,
                type='expert_subGraph',
                member_count=len(members),
                members=members,
                keywords=keywords,
                description=description,
                node_id=detail.get('node_id'),
                auto_discovered=detail.get('auto_discovered', False),
                metadata=detail
            )

        logger.info(f"✅ Indices built successfully:")
        logger.info(f"   - {len(self.shared_subGraphs)} shared subGraphs")
        logger.info(f"   - {len(self.expert_subGraphs)} expert subGraphs")
        logger.info(f"   - {len(self.node_memberships_map)} nodes with memberships")

    def get_shared_subGraphs(self) -> Dict[str, SubGraphInfo]:
        return self.shared_subGraphs

    def get_expert_subGraphs(self) -> Dict[str, SubGraphInfo]:
        return self.expert_subGraphs


    def get_statistics(self) -> Dict[str, Any]:
        total_shared_members = sum(c.member_count for c in self.shared_subGraphs.values())
        total_expert_members = sum(c.member_count for c in self.expert_subGraphs.values())

        return {
            'total_subGraphs': len(self.shared_subGraphs) + len(self.expert_subGraphs),
            'shared_subGraphs': len(self.shared_subGraphs),
            'expert_subGraphs': len(self.expert_subGraphs),
            'nodes_with_memberships': len(self.node_memberships_map),
            'total_shared_members': total_shared_members,
            'total_expert_members': total_expert_members,
            'avg_shared_subGraph_size': total_shared_members / len(self.shared_subGraphs) if self.shared_subGraphs else 0,
            'avg_expert_subGraph_size': total_expert_members / len(self.expert_subGraphs) if self.expert_subGraphs else 0
        }
