
import time
import psutil
import os
from ..utils import logger, LLMCompletionCall

from .retrieval_utils import ResultSaver


CHAIN_REASONING_HEAD = """You are a factual question answering assistant. Answer the question based on the provided context and previous reasoning steps.

CRITICAL INSTRUCTIONS:
1. You MUST provide a direct answer to the question, no matter what
2. If the context is helpful, use it to answer
3. If the context is not relevant or insufficient, use your own knowledge to provide the best possible answer
4. NEVER say things like "the context doesn't provide", "information is not available", "I cannot answer", etc.
5. Output ONLY the answer itself - no meta-commentary, no explanations about the context"""

CHAIN_REASONING_INSTRUCTION = """Output only the answer. No explanations, reasoning, or additional text. Be precise and concise. If the context is not relevant, answer based on your knowledge."""

CHAIN_FINAL_HEAD = """You are a factual question answering assistant.
Based on the comprehensive knowledge context accumulated through iterative retrieval, provide the final answer to the Original Question."""

CHAIN_FINAL_INSTRUCTION = """Output ONLY the final answer to the Original Question based on the sub-questions and retrieved information.
Do not include any reasoning process, sub-question answers, or explanations.
Provide a comprehensive and accurate final answer:"""


class MixtureOfGraphQuestionAnswerPipeline:
    def __init__(self, config):
        self.llm_client = LLMCompletionCall(config.output.results_dir)
        self.config=config


    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB

    def combine_retrieved_chunks_and_triples(self, chunk_contents, triple_contents, triple_paths):
        context = "=== Retrieved Knowledge ==="
        triple_strings = ['({}, {}, {})'.format(*triple) for triple in triple_contents]
        final_triple_content = ', '.join(triple_strings)
        context += f"\nTriples:\n{final_triple_content}\n"
        if triple_paths:
            context += "\nTriple Paths:\n"
            for path_str in triple_paths:
                context += f"{path_str}\n"

        context += f"Chunks:"
        for i, chunk_content in enumerate(chunk_contents):
            context += f"\nChunk {i+1}: {chunk_content}"
        return context

    def retrieval_with_mog(self, MoG_retriever, retrieval_query, top_k, processing_stats):

        retrieval_start = time.time()
        mog_result = MoG_retriever.retrieve(retrieval_query, top_k=top_k)
        retrieval_time = time.time() - retrieval_start

        chunk_contents = []
        for chunk_id in mog_result.final_chunk_ids:
            sub_content = MoG_retriever.chunk_id2text.get(chunk_id, f"[Missing content for chunk {chunk_id}]")
            chunk_contents.append(sub_content)
        triple_contents = mog_result.final_triples
        retrieved_triple_paths = mog_result.final_triple_paths
        logger.warning(f" MoGretrieval completed in {retrieval_time:.2f}s")
        logger.info(f"   Retrieved {len(mog_result.final_chunk_ids)} chunks")
        logger.info(f"   Retrieved {len(triple_contents)} triples")
        logger.info(f"   Retrieved {len(retrieved_triple_paths)} triple paths")
        logger.warning(f"   Activated experts: {len(mog_result.activated_experts)}")
        triple_strings = ['({}, {}, {})'.format(*triple) for triple in triple_contents]
        processing_stats["activated_experts"].extend(mog_result.activated_experts)
        processing_stats["retrieved_chunks"].extend(mog_result.final_chunk_ids)
        processing_stats["retrieved_chunk_contents"].extend(chunk_contents)
        processing_stats["retrieved_triples"].extend(triple_strings)
        processing_stats["retrieved_triple_paths"].extend(retrieved_triple_paths)

        return chunk_contents, triple_strings, retrieved_triple_paths, processing_stats, retrieval_time

    def iterative_subquestions_processing(self, MoG_retriever, sub_questions, top_k, original_question, retrieval_mode=""):

        sub_answers = []
        previous_sub_question= []
        previous_answers = []
        processing_stats_for_all_sub_questions = []
        retrieval_time_sub_query_list = []
        generation_time_sub_answer_list = []

        for i, subq in enumerate(sub_questions):
            processing_stats = {
                "activated_experts": [],
                "retrieved_chunks": [],
                "retrieved_chunk_contents": [],
                "retrieved_triples": [],
                "retrieved_triple_paths": []
            }
            if isinstance(subq, dict) and "sub-question" in subq:
                sub_question = subq["sub-question"]
            elif isinstance(subq, str):
                sub_question = subq
            elif isinstance(subq, list):
                if subq and isinstance(subq[0], dict) and "sub-question" in subq[0]:
                    sub_question = subq[0]["sub-question"]
                elif subq and isinstance(subq[0], str):
                    sub_question = subq[0]
                else:
                    logger.error(f"Unexpected subq type at index {i}: {type(subq)}, use original question instead.")
                    sub_question = original_question
            else:
                logger.error(f"Unexpected subq type at index {i}: {type(subq)}, use original question instead.")
                sub_question = original_question
            logger.info(f"   Processing sub-question {i+1}/{len(sub_questions)}: {sub_question}")

            if i > 0 and previous_answers:
                prev_ans = previous_answers[-1]
                retrieval_query = prev_ans + ", " + sub_question
            else:
                retrieval_query = sub_question

            logger.warning(f"   Retrieval query of sub-question {i+1}/{len(sub_questions)}: {retrieval_query}")
            chunk_contents, triple_contents, triple_paths, processing_stats, retrieval_time_sub_query = self.retrieval_with_mog(MoG_retriever, retrieval_query, top_k, processing_stats)
            processing_stats_for_all_sub_questions.append(processing_stats)
            retrieval_time_sub_query_list.append(round(retrieval_time_sub_query, 3))

            context = self.combine_retrieved_chunks_and_triples(chunk_contents, triple_contents, triple_paths)
            if not previous_answers:
                qa_prompt = (
                    f"{CHAIN_REASONING_HEAD}\n"
                    f"Context: {context}\n"
                    f"Sub-question: {sub_question}\n"
                    f"{CHAIN_REASONING_INSTRUCTION}"
                )
            else:
                previous_context = "\n".join([
                    f"Previous question {i + 1}: {q}\nPrevious answer {i + 1}: {a}"
                    for i, (q, a) in enumerate(zip(previous_sub_question, previous_answers))
                ])
                qa_prompt = (
                    f"{CHAIN_REASONING_HEAD}\n"
                    f"Context: {context}\n"
                    f"{previous_context}\n"
                    f"Sub-question: {sub_question}\n"
                    f"{CHAIN_REASONING_INSTRUCTION}"
                )

            retrieval_start_sub_answer_generation = time.time()
            answer = self.llm_client.call_api(qa_prompt).strip()
            generation_time_sub_answer = time.time() - retrieval_start_sub_answer_generation
            generation_time_sub_answer_list.append(round(generation_time_sub_answer, 3))
            previous_answers.append(answer)
            previous_sub_question.append(sub_question)
            logger.warning(f"   Answer of sub-question {i+1}/{len(sub_questions)} {sub_question} : {answer}")

            sub_answers.append({
                "sub_question": sub_question,
                "retrieved": context,
                "answer": answer,
                "step": i + 1
            })

        return sub_answers, processing_stats_for_all_sub_questions, retrieval_time_sub_query_list, generation_time_sub_answer_list

    def final_query_processing(self, MoG_retriever, sub_answers, original_question, top_k, processing_stats_sub_questions, retrieval_mode):

        subq_text = "Initial analysis: " + "\n".join([
            f"Sub-question {item['step']}: {item['sub_question']}\n"
            f"Sub-answer: {item['answer']}\n"
            f"Retrieved: {item['retrieved']}\n"
            for item in sub_answers
        ])

        final_prompt = (
            f"{CHAIN_FINAL_HEAD}\n\n"
            f"Original Question: {original_question}\n{subq_text}\n"
            f"{CHAIN_REASONING_HEAD}\n\n"
            f"Original Question: {original_question}\n"
            f"{CHAIN_FINAL_INSTRUCTION}\n"
        )

        retrieval_start_final_answer_generation = time.time()

        inited_answer = self.llm_client.call_api(final_prompt).strip()
        generation_final_answer_time = time.time() - retrieval_start_final_answer_generation

        if "irCoT-5" in retrieval_mode:
            logger.warning("Using IRCoT with 5 max steps for final query processing.")
            max_steps = 5
        elif "irCoT-3" in retrieval_mode:
            logger.warning("Using IRCoT with 3 max steps for final query processing.")
            max_steps = 3
        elif "irCoT-1" in retrieval_mode:
            logger.warning("Using IRCoT with 3 max steps for final query processing.")
            max_steps = 1
        else:
            logger.warning("Not using IRCoT for final query processing.")
            max_steps = 0
        step = 1
        if max_steps !=0:
            subq_qa = "Initial analysis" + "\n".join([
                f"Sub-question {item['step']}: {item['sub_question']}\n"
                f"Sub-answer: {item['answer']}\n"
                for item in sub_answers
            ])

            retrieved_chunk_contents = list(set(processing_stats_sub_questions["retrieved_chunk_contents"]))
            retrieved_triples = list(set(processing_stats_sub_questions["retrieved_triples"]))
            retrieved_triple_paths = list(set(processing_stats_sub_questions["retrieved_triple_paths"]))

            inited_thought = f"Original question: {original_question} " + subq_qa + f" Initial answer: {inited_answer}"

            logger.info(f"🚀 Starting IRCoT for question: {original_question}")
            thoughts = [inited_thought]
            current_query = original_question
            processing_stats_ircot = []
            while step <= max_steps:
                logger.warning(f"📝 IRCoT Step {step}/{max_steps}")
                step += 1
                context = "=== Triples ===\n" + "\n".join(retrieved_triples)
                if retrieved_triple_paths:
                    context += "\n=== Triple Paths ===\n" + "\n".join(retrieved_triple_paths)
                context += "\n=== Chunks ===\n" + "\n".join(retrieved_chunk_contents)

                ircot_prompt = f"""
                                You are an expert knowledge assistant using iterative retrieval with chain-of-thought reasoning.
    
                                Current Question: {current_query}
    
                                Available Knowledge Context:
                                {context}
    
                                Previous Thoughts: {' | '.join(thoughts) if thoughts else 'None'}
    
                                Step {step}: Please think step by step about what additional information you need to answer the question completely and accurately.
    
                                Instructions:
                                1. Analyze the current knowledge context and the question
                                2. Consider the initial analysis from sub-question processing (if available)
                                3. Think about what information might be missing or unclear
                                4. If you have enough information to answer, in the end of your response, write "So the answer is:" followed by your final answer
                                5. If you need more information, in the end of your response, write a specific query begin with "The new query is:" to retrieve additional relevant information
                                6. Be specific and focused in your reasoning
                                7. Build upon the initial analysis to provide deeper insights
    
                                Your reasoning:
                                """
                response = self.llm_client.call_api(ircot_prompt).strip()

                thoughts.append(response)

                logger.info(f"Step {step} response: {response[:100]}...")

                if "So the answer is:" in response:
                    logger.warning(" Final answer found, stopping IRCoT")
                    break

                if "The new query is:" in response:
                    new_query = response.split("The new query is:")[1].strip()
                else:
                    new_query = response

                if new_query and new_query != current_query:
                    current_query = new_query
                    logger.warning(f"🔄 New query for next iteration: {current_query}")

                    processing_stats = {
                        "activated_experts": [],
                        "retrieved_chunks": [],
                        "retrieved_chunk_contents": [],
                        "retrieved_triples": [],
                        "retrieved_triple_paths": []
                    }

                    chunk_contents, triple_contents, triple_paths, processing_stats, retrieval_time_ircot_query= self.retrieval_with_mog(MoG_retriever,
                                                                                                              current_query,
                                                                                                              top_k,
                                                                                                              processing_stats)
                    processing_stats_ircot.append(processing_stats)

                    retrieved_chunk_contents.extend(chunk_contents)
                    retrieved_triples.extend(triple_contents)
                    retrieved_triple_paths.extend(triple_paths)

                    retrieved_chunk_contents = list(set(retrieved_chunk_contents))
                    retrieved_triples = list(set(retrieved_triples))
                    retrieved_triple_paths = list(set(retrieved_triple_paths))


                    logger.info(f"Retrieved {len(triple_contents)} new triples, {len(chunk_contents)} new chunks, {len(triple_paths)} new triple paths")
                else:
                    logger.info("No new query generated, stopping IRCoT")
                    break


            final_context = "=== Triples ===\n" + "\n".join(retrieved_triples)
            if retrieved_triple_paths:
                final_context += "\n=== Triple Paths ===\n" + "\n".join(retrieved_triple_paths)
            final_context += "\n=== Chunks ===\n" + "\n".join(retrieved_chunk_contents)

            final_prompt = f"""
                            You are an expert knowledge assistant. Your task is to answer the question based on the provided knowledge context.
    
                            1. Use ONLY the information from the provided knowledge context and try your best to answer the question.
                            2. If the context is not relevant or insufficient, use your own knowledge to provide the best possible answer
                            3. Be precise and concise in your answer
                            4. For factual questions, provide the specific fact or entity name
                            5. For temporal questions, provide the specific date, year, or time period
    
                            Question: {original_question}
    
                            Knowledge Context:
                            {final_context}
                            
                            Previous Thoughts: {' | '.join(thoughts) if thoughts else 'None'}
                            Now, answer the question: {original_question} (be specific and direct):
                            """
            final_answer = self.llm_client.call_api(final_prompt).strip()
        else:
            final_answer = inited_answer
            processing_stats_ircot = []

        logger.info(f"   Answer of original question {original_question} : {final_answer}")
        return final_answer, processing_stats_ircot, step-1, generation_final_answer_time

    def mixture_of_graph_question_answer(self, config, graph_question_decomposer, MoG_retriever, qa_pairs, schema_path, resume_from=None, retrieval_mode=""):
        task_start_time = 0
        accuracy = 0
        total_questions = len(qa_pairs)
        dataset_name = MoG_retriever.dataset_name
        result_saver = ResultSaver(
            dataset_name=dataset_name,
            algorithm="MoG",
            mode="agent",
            resume_from=resume_from,
            results_dir=self.config.output.results_dir,
            retrieval_mode = retrieval_mode,
            top_k = self.config.retrieval.top_k
        )

        logger.warning("🤖 Starting MoG Agent Retrieval Mode (IRCoT)")
        logger.warning(f"Processing {total_questions} questions with iterative retrieval")
        logger.warning("-" * 80)


        skipped_count = 0
        processed_count = 0

        for idx, qa in enumerate(qa_pairs):
            question_id = f"{dataset_name}_{idx}"

            # Skip if already processed
            if result_saver.is_processed(question_id):
                logger.warning(f"\n{'='*80}")
                logger.warning(f"⏭️  Skipping Question {dataset_name}_{idx} in {total_questions} questions (already processed)")
                logger.warning(f"{'='*80}")
                skipped_count += 1
                continue

            original_question = qa["question"]
            processed_count += 1
            logger.warning(f"\n{'='*80}")
            logger.warning(f"🔍 Processing Question {dataset_name}_{idx} in {total_questions} questions")
            logger.warning(f"{'='*80}")
            logger.warning(f"Question: {original_question}")

            question_start_time = time.time()

            logger.info("📝 Step 1: Decomposing original question into several sub-questions")

            decomposition_start_time = time.time()

            sub_questions = graph_question_decomposer.question_decomposition(
                original_question,
                schema_path
            )

            decomposition_time = time.time() - decomposition_start_time
            logger.info("📝 Step 2: Iterative retrieval and answering of sub-questions with MoG")

            logger.info(f"📊 Memory before Step 2 of Question {idx + 1}: {self.get_memory_usage():.2f} MB")

            sub_query_start_time = time.time()

            logger.warning("Using iterative sub-question processing mode.")
            sub_answers, processing_stats_for_all_sub_questions, retrieval_time_sub_query_list, generation_time_sub_answer_list= self.iterative_subquestions_processing(
                MoG_retriever, sub_questions, config.retrieval.top_k, original_question, retrieval_mode)


            sub_query_time = time.time() - sub_query_start_time
            processing_stats = {
                "activated_experts": [],
                "retrieved_chunks": [],
                "retrieved_chunk_contents": [],
                "retrieved_triples": [],
                "retrieved_triple_paths": []
            }
            for stats in processing_stats_for_all_sub_questions:
                processing_stats["activated_experts"].extend(stats["activated_experts"])
                processing_stats["retrieved_chunks"].extend(stats["retrieved_chunks"])
                processing_stats["retrieved_chunk_contents"].extend(stats["retrieved_chunk_contents"])
                processing_stats["retrieved_triples"].extend(stats["retrieved_triples"])
                processing_stats["retrieved_triple_paths"].extend(stats["retrieved_triple_paths"])

            final_answer, processing_stats_ircot, ircotstep, generation_final_answer_time = self.final_query_processing(
                MoG_retriever, sub_answers, original_question, config.retrieval.top_k, processing_stats, retrieval_mode)

            question_time = time.time() - question_start_time

            logger.warning(f"\n{'='*80}")
            logger.warning(f"🎉 Question {idx} completed in {question_time:.2f}s")
            logger.warning(f"🎯 ircot step: {ircotstep}")
            logger.warning(f"{'='*80}")
            logger.warning(f"❓ Question: {original_question}")
            logger.warning(f"💡 Generated Answer: {final_answer}")
            logger.warning(f"🎯 Gold Answer: {qa['answer']}")
            logger.warning("-" * 80)

            eval_result = self.llm_client.eval(original_question, qa["answer"], final_answer)
            logger.info(f"📋 MoGAgent eval result: {eval_result}")

            if eval_result == "1":
                accuracy += 1

            logger.warning(f"Eval result: {'Correct ' if eval_result == '1' else 'Wrong ❌'}")


            for stats_list in [processing_stats_ircot, processing_stats_for_all_sub_questions]:
                for stats in stats_list:
                    processing_stats["activated_experts"].extend(stats["activated_experts"])
                    processing_stats["retrieved_chunks"].extend(stats["retrieved_chunks"])
                    processing_stats["retrieved_chunk_contents"].extend(stats["retrieved_chunk_contents"])
                    processing_stats["retrieved_triples"].extend(stats["retrieved_triples"])
                    processing_stats["retrieved_triple_paths"].extend(stats["retrieved_triple_paths"])

            activated_experts = list(set(processing_stats["activated_experts"]))
            retrieved_chunks = list(set(processing_stats["retrieved_chunks"]))
            retrieved_triples = list(set(processing_stats["retrieved_triples"]))
            retrieved_triple_paths = list(set(processing_stats["retrieved_triple_paths"]))

            result_data = {
                "correct": 1 if eval_result == "1" else 0,
                "question_id": f"{dataset_name}_{idx}",
                "ircotstep": ircotstep,
                "question": original_question,
                "ground_truth": qa["answer"],
                "generated_answer": final_answer,
                "processing_time": round(question_time, 3),
                "decomposition_time": round(decomposition_time, 3),
                "sub_query_time": round(sub_query_time, 3),
                "generation_final_answer_time": round(generation_final_answer_time, 3),
                "sub_query_num": len(sub_answers),
                "retrieval_time_sub_query":retrieval_time_sub_query_list,
                "retrieval_time_sub_query_avg": round(sum(retrieval_time_sub_query_list)/len(retrieval_time_sub_query_list),3),
                "generation_time_sub_answer":generation_time_sub_answer_list,
                "generation_time_sub_answer_avg":round(sum(generation_time_sub_answer_list)/len(generation_time_sub_answer_list),3),
                "activated_experts_count": len(activated_experts),
                "activated_experts": activated_experts,
                "retrieved_triples_count": len(retrieved_triples),
                "retrieved_chunks_count": len(retrieved_chunks),
            }

            result_saver.save_result(result_data)

            retrieval_results = {
                "correct": 1 if eval_result == "1" else 0,
                "question_id": f"{dataset_name}_{idx}",
                "ircotstep": ircotstep,
                "question": qa['question'],
                "ground_truth": qa["answer"],
                "generated_answer": final_answer,
                "processing_time": round(question_time, 3),
                "retrieved_triples_count": len(retrieved_triples),
                "retrieved_chunk_contents": list(set(processing_stats["retrieved_chunk_contents"])),
                "retrieved_triples": retrieved_triples,
                "retrieved_chunks": retrieved_chunks,
                "retrieved_triple_paths_count": len(retrieved_triple_paths),
                "retrieved_triple_paths": retrieved_triple_paths
            }

            result_saver.save_retrieval_data(retrieval_results)




        total_time = time.time() - task_start_time
        avg_time = total_time / total_questions

        total_hours = int(total_time // 3600)
        total_minutes = int((total_time % 3600) // 60)
        total_seconds = int(total_time % 60)

        avg_hours = int(avg_time // 3600)
        avg_minutes = int((avg_time % 3600) // 60)
        avg_seconds = int(avg_time % 60)

        logger.warning(f"\n{'='*80}")
        logger.warning(f"FINAL RESULTS - MoGAgent Mode")
        logger.warning(f"{'='*80}")
        logger.warning(f"Overall Accuracy: {accuracy / total_questions * 100:.2f}% ({accuracy}/{total_questions})")
        logger.warning(f"Average time taken: {avg_hours}h{avg_minutes}m{avg_seconds}s")
        logger.warning(f"Total time: {total_hours}h{total_minutes}m{total_seconds}s")
        logger.warning(f"{'='*80}")

        result_saver.finalize(
            accuracy=accuracy,
            total_time=total_time,
            total_questions=total_questions,
            additional_stats={
                "schema_path": schema_path,
                "config": {
                    "top_k": config.retrieval.top_k if hasattr(config.retrieval, 'top_k') else None
                }
            }
        )

