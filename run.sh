#!/bin/bash

# ============================================================
# MoG
# Pipeline execution script
# Usage: Uncomment the commands for the dataset/stage you need.
# ============================================================

# -------------------- demo --------------------

# Stage 1: Knowledge Graph Preprocessing
# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": true, "retrieve_trigger": false}}' \
#   --datasets demo --construction_mode KGPreprocess

# Stage 2: MoG Graph Building
# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": true, "retrieve_trigger": false}}' \
#   --datasets demo --construction_mode MoGBuild

# Stage 3: MoG Retrieval
# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": false, "retrieve_trigger": true}}' \
#   --datasets demo --retrieval_mode MoGRetrieval --construction_mode MoGBuild

# Stage 4: MoG Retrieval with IRCoT
# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": false, "retrieve_trigger": true}}' \
#   --datasets demo --retrieval_mode MoGRetrieval_irCoT-5 --construction_mode MoGBuild

# -------------------- 2wiki --------------------

# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": true, "retrieve_trigger": false}}' \
#   --datasets 2wiki --construction_mode KGPreprocess

# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": true, "retrieve_trigger": false}}' \
#   --datasets 2wiki --construction_mode MoGBuild

# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": false, "retrieve_trigger": true}}' \
#   --datasets 2wiki --retrieval_mode MoGRetrieval --construction_mode MoGBuild

# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": false, "retrieve_trigger": true}}' \
#   --datasets 2wiki --retrieval_mode MoGRetrieval_irCoT-5 --construction_mode MoGBuild

# -------------------- hotpot --------------------

# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": true, "retrieve_trigger": false}}' \
#   --datasets hotpot --construction_mode KGPreprocess

# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": true, "retrieve_trigger": false}}' \
#   --datasets hotpot --construction_mode MoGBuild

# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": false, "retrieve_trigger": true}}' \
#   --datasets hotpot --retrieval_mode MoGRetrieval --construction_mode MoGBuild

# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": false, "retrieve_trigger": true}}' \
#   --datasets hotpot --retrieval_mode MoGRetrieval_irCoT-5 --construction_mode MoGBuild

# -------------------- musique --------------------

# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": true, "retrieve_trigger": false}}' \
#   --datasets musique --construction_mode KGPreprocess

# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": true, "retrieve_trigger": false}}' \
#   --datasets musique --construction_mode MoGBuild

# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": false, "retrieve_trigger": true}}' \
#   --datasets musique --retrieval_mode MoGRetrieval --construction_mode MoGBuild

# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": false, "retrieve_trigger": true}}' \
#   --datasets musique --retrieval_mode MoGRetrieval_irCoT-5 --construction_mode MoGBuild

# -------------------- graphrag-bench --------------------


# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": true, "retrieve_trigger": false}}' \
#   --datasets graphrag-bench --construction_mode KGPreprocess

# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": true, "retrieve_trigger": false}}' \
#   --datasets graphrag-bench --construction_mode MoGBuild

# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": false, "retrieve_trigger": true}}' \
#   --datasets graphrag-bench --retrieval_mode MoGRetrieval --construction_mode MoGBuild

# python main.py --config config/mog_config.yaml --override '{"triggers": {"constructor_trigger": false, "retrieve_trigger": true}}' \
#   --datasets graphrag-bench --retrieval_mode MoGRetrieval_irCoT-5 --construction_mode MoGBuild
