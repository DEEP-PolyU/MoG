
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Set
import re
from ...utils import logger


class ResultSaver:
    def __init__(self, dataset_name: str, algorithm: str, mode: str, results_dir: str = "results",
                 resume_from: Optional[str] = None, retrieval_mode="",
                 top_k=50):
        self.dataset_name = dataset_name
        self.algorithm = algorithm
        self.mode = mode
        self.results_dir = results_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        dataset_dir = Path(results_dir) / dataset_name / retrieval_mode
        dataset_dir.mkdir(parents=True, exist_ok=True)

        self.result_file = dataset_dir / f"{self.dataset_name}_{algorithm}_topK-{top_k}_{self.timestamp}_results.jsonl"

        if resume_from is None:
            filename_pattern = f"{self.dataset_name}_{self.algorithm}_topK-{top_k}_*_results.jsonl"

            matching_files = list(dataset_dir.glob(filename_pattern))
            if matching_files:
                matching_files.sort(key=lambda x: x.stem.split('_')[-2],
                                    reverse=True)

                latest_file = matching_files[0]

                current_filename_pattern = f"{self.dataset_name}_{self.algorithm}_topK-{top_k}_{self.timestamp}_results.jsonl"
                if latest_file.name != current_filename_pattern:
                    resume_from = str(latest_file)
                    print(resume_from)
                    logger.warning(f"📂 Auto-detected previous run file for resuming: {latest_file.name}")

        retrieved_contents_dir = Path(results_dir) / dataset_name / retrieval_mode / "retrieved_contents"
        retrieved_contents_dir.mkdir(parents=True, exist_ok=True)
        self.retrieval_file = retrieved_contents_dir / f"{self.dataset_name}_{algorithm}_topK-{top_k}_{self.timestamp}_retrieval.jsonl"

        self.saved_count = 0
        self.processed_question_ids: Set[str] = set()

        logger.info(f"📁 ResultSaver initialized")
        logger.info(f"   - Dataset: {dataset_name}")
        logger.info(f"   - Algorithm: {algorithm}")
        logger.info(f"   - Mode: {mode}")
        if resume_from:
            logger.warning(f"   - Resuming from: {resume_from}")
        logger.info(f"   - Result file: {self.result_file}")

        # Resume from previous results if specified
        if resume_from:
            self._resume_from_file(resume_from)

    def save_result(self, result_data: Dict[str, Any]) -> bool:
        try:
            result_data.pop('timestamp', None)

            with open(self.result_file, 'a', encoding='utf-8') as f:
                json.dump(result_data, f, ensure_ascii=False)
                f.write('\n')

            self.saved_count += 1

            logger.info(f"💾 Saved result for {result_data.get('question_id')}")
            logger.debug(f"   - Correct: {result_data.get('correct', -1)}")
            logger.debug(f"   - Total saved: {self.saved_count}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to save result: {str(e)}")
            return False

    def save_retrieval_data(self, retrieval_data: Dict[str, Any]) -> bool:
        try:
            retrieval_data.pop('timestamp', None)

            with open(self.retrieval_file, 'a', encoding='utf-8') as f:
                json.dump(retrieval_data, f, ensure_ascii=False)
                f.write('\n')


            logger.info(f"💾 Saved retrieval data for {retrieval_data.get('question_id')}")
            logger.debug(f"   - Correct: {retrieval_data.get('correct', -1)}")
            logger.debug(f"   - Total saved: {self.saved_count}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to save result: {str(e)}")
            return False

    def transform_path_original(self, path):
        path_obj = Path(path)

        filename = path_obj.name

        new_filename = filename.replace("_results.jsonl", "_retrieval.jsonl")
        new_path = path_obj.parent / "retrieved_contents" / new_filename

        return str(new_path)

    def _resume_from_file(self, resume_file: str) -> None:
        resume_path = Path(resume_file)
        resume_retrieval_file = self.transform_path_original(resume_file)
        resume_retrieval_path = Path(resume_retrieval_file)

        if not resume_path.exists() or not resume_retrieval_path.exists():
            logger.warning(f"⚠️  Resume file not found: {resume_file}")
            return

        logger.info(f"🔄 Resuming from: {resume_file}")

        try:
            previous_results = []
            with open(resume_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get('type') == 'summary':
                            continue

                        previous_results.append(data)

                        if 'question_id' in data:
                            self.processed_question_ids.add(data['question_id'])

                    except json.JSONDecodeError as e:
                        logger.warning(f"   Skipping invalid JSON line: {e}")
                        continue

            previous_retrieval = []
            with open(resume_retrieval_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        previous_retrieval.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"   Skipping invalid JSON line: {e}")
                        continue

            if previous_results:
                with open(self.result_file, 'w', encoding='utf-8') as f:
                    for result in previous_results:
                        json.dump(result, f, ensure_ascii=False)
                        f.write('\n')

                self.saved_count = len(previous_results)

                logger.warning(f"✅ Loaded {len(previous_results)} previous results")
                logger.warning(f"   - Processed questions: {len(self.processed_question_ids)}")
                logger.warning(f"   - Will skip these questions and continue from where left off")
            else:
                logger.warning(f"   No valid results found in resume file")


            if previous_retrieval:
                with open(self.retrieval_file, 'w', encoding='utf-8') as f:
                    for retrieval in previous_retrieval:
                        json.dump(retrieval, f, ensure_ascii=False)
                        f.write('\n')
                logger.warning(f"✅ Loaded {len(previous_retrieval)} previous retrieval data")

        except Exception as e:
            logger.error(f"❌ Failed to resume from file: {e}")
            import traceback
            traceback.print_exc()

    def is_processed(self, question_id: str) -> bool:
        return question_id in self.processed_question_ids

    def finalize(self, accuracy: int, total_time: float, total_questions: int,
                 additional_stats: Optional[Dict[str, Any]] = None) -> bool:
        try:
            accuracy_pct = round(accuracy / total_questions * 100, 2) if total_questions > 0 else 0.0

            logger.info(f"✅ Finalized results file: {self.result_file}")
            logger.info(f"   - Total saved: {self.saved_count} questions")
            logger.info(f"   - Accuracy: {accuracy_pct}%")
            logger.info(f"   - Total time: {round(total_time, 2)}s")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to finalize results: {str(e)}")
            return False

    def get_processed_question_ids(self) -> Set[str]:
        processed_ids = set()

        if not self.result_file.exists():
            return processed_ids

        try:
            with open(self.result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if 'question_id' in data and data.get('type') != 'summary':
                            processed_ids.add(data['question_id'])
                    except json.JSONDecodeError:
                        continue

            logger.info(f"🔄 Found {len(processed_ids)} previously processed questions")

        except Exception as e:
            logger.warning(f"⚠️ Could not read existing results: {str(e)}")

        return processed_ids

    def get_result_file_path(self) -> str:
        return str(self.result_file)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset_name,
            "algorithm": self.algorithm,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "saved_count": self.saved_count,
            "result_file": str(self.result_file)
        }

