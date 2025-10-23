import os
import json
import glob
import argparse
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, model_validator
from openai import OpenAI
from tqdm.auto import tqdm
from datetime import datetime

# Sub-question weights for scoring multi-part answers.
SUB_QUESTION_WEIGHTS = {2: [0.4348, 0.5652], 3: [0.2506, 0.3258, 0.4236], 4: [0.1616, 0.2101, 0.2732, 0.3551]}

# --- Pydantic Model for API Response ---
class EvaluationResult(BaseModel):
    analysis: str = Field(description="An explanation of the extraction and judgment process for the prediction.")
    gt_answers: List[str] = Field(description="A list of parsed answers from the ground truth.")
    pred_answers: List[Optional[str]] = Field(description="A list of parsed/extracted answers from the prediction.")
    correctness: List[bool] = Field(description="A boolean list indicating if each predicted answer part is correct.")

    @model_validator(mode='after')
    def check_list_lengths(self) -> 'EvaluationResult':
        len_gt = len(self.gt_answers)
        len_pred = len(self.pred_answers)
        len_corr = len(self.correctness)
        if not (len_gt > 0 and len_gt == len_pred == len_corr):
            raise ValueError(
                f"List lengths must be equal and non-empty. Got: "
                f"gt_answers={self.gt_answers} with length {len_gt}, "
                f"pred_answers={self.pred_answers} with length {len_pred}, "
                f"correctness={self.correctness} with length {len_corr}."
            )
        return self

def timestamp() -> str:
    dt_string = datetime.now().strftime("%m-%d_%H-%M-%S")
    return dt_string  

# --- Utility Functions ---
def load_json(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"Warning: Could not load or parse JSON from {file_path}. Error: {e}")
        return None

def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# --- Main Evaluator Class ---
class MathAnswerEvaluator:
    def __init__(self, args):
        self.args = args
        self.client = OpenAI(api_key=args.api_key, base_url=args.base_url)
        
        with open(args.prompt_file, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()
        
        self.model_name = self.args.model.replace('/', '__')

    def find_items_to_process(self) -> List[Dict[str, str]]:
        items_to_process = []
        all_result_dirs = glob.glob(os.path.join(self.args.inference_dir, "*"))
        
        print(f"Scanning {len(all_result_dirs)} potential result directories in {self.args.inference_dir}...")
    
        for result_dir in all_result_dirs:
            if not os.path.isdir(result_dir):
                continue

            item = {
                "id": os.path.basename(result_dir),
                "path": result_dir,
                "ori_json_path": os.path.join(result_dir, "ori.json"),
                "pred_json_path": os.path.join(result_dir, "reasoning_result.json"),
                "output_path": os.path.join(result_dir, f"{self.model_name}--evaluation_result.json"),
            }
    
            if not (os.path.exists(item["ori_json_path"]) and os.path.exists(item["pred_json_path"])):
                continue
    
            should_process = False
            if self.args.force_rerun:
                should_process = True
            else:
                if not os.path.exists(item["output_path"]):
                    should_process = True
                else:
                    try:
                        eval_data = load_json(item["output_path"])
                        if eval_data is None or eval_data.get("status") == "error" or "evaluation" not in eval_data:
                            print(f"Item {item['id']} has a previous error or invalid result. Scheduling for re-run.")
                            should_process = True
                    except Exception as e:
                        print(f"Warning: Corrupted output file {item['output_path']} ({e}). Scheduling for re-run.")
                        should_process = True
            
            if should_process:
                items_to_process.append(item)
        
        return items_to_process

    def format_input_for_api(self, item: Dict[str, str]) -> Optional[str]:
        """Loads data and formats it into a JSON string for the API prompt."""
        ori_data = load_json(item["ori_json_path"])
        pred_data = load_json(item["pred_json_path"])

        if not ori_data or not pred_data:
            return None

        reasoning_steps = pred_data.get("reasoning_steps", [])
        predicted_texts = []
        for step in reasoning_steps:
            if step.get('type') in ['text']:
                predicted_texts.append(step.get('content', ''))
        
        prediction_solution = "\n".join(predicted_texts).strip()

        input_dict = {
            "question_text": ori_data.get("question", ""),
            "ground_truth_answer": ori_data.get("answer", ""),
            "prediction_solution": prediction_solution,
        }

        return json.dumps(input_dict, ensure_ascii=False, indent=2)

    def call_api_openai_schema(self, input_data_str: str) -> Dict:
        """Calls OpenAI API with structured output parsing."""
        messages = [{"role": "user", "content": self.prompt_template.format(input_data=input_data_str)}]
        
        parameters = {
            'model': self.args.model,
            'messages': messages,
            'response_format': EvaluationResult,
            'max_completion_tokens': self.args.max_tokens,
            'temperature': self.args.temperature,
            'reasoning_effort': self.args.reasoning_effort
        }

        if 'gpt-4.1' in self.args.model:
            del parameters['reasoning_effort']
        
        response = self.client.beta.chat.completions.parse(**parameters)
        api_result = response.choices[0].message.parsed
        return api_result.model_dump()

    def process_single_item(self, item: Dict[str, str]) -> Dict:
        """Processes one item: formats input, calls API, and returns result."""
        try:
            input_data_str = self.format_input_for_api(item)
            if not input_data_str:
                raise ValueError("Failed to format input data for API.")

            # Simplified: Always call the OpenAI schema method
            api_result = self.call_api_openai_schema(input_data_str)

            EvaluationResult.model_validate(api_result)

            save_result = {"id": item["id"], "status": "success", "evaluation": api_result}
            save_json(save_result, item["output_path"])
            return save_result

        except Exception as e:
            error_info = {"id": item["id"], "status": "error", "reason": str(e)}
            print(f"API call failed for item {item['id']}: {e}")
            traceback.print_exc()
            save_json(error_info, item["output_path"])
            return error_info

    def _update_stats(self, stats_dict: Dict, key: str, score: float):
        """Simplified helper function to update statistics."""
        if key not in stats_dict:
            stats_dict[key] = {'count': 0, 'total_score': 0.0}
        stats_dict[key]['count'] += 1
        stats_dict[key]['total_score'] += score

    def summarize_results(self, all_items: List[Dict]):
        """
        Creates a simplified summary of evaluation results, focusing on overall, 
        knowledge, and image presence metrics.
        """
        # Simplified stats structure
        stats = {
            'overall': {'count': 0, 'total_score': 0.0, 'completely_correct_count': 0},
            'by_knowledge': {},
            'by_question_image_count': {},
        }
        error_count = 0
        too_many_parts_count = 0
        all_results_data = []

        for item in tqdm(all_items, desc="Summarizing results"):
            eval_result = load_json(item['output_path'])
            ori_data = load_json(item['ori_json_path'])
            all_results_data.append({"id": item['id'], **(eval_result or {})})

            if not eval_result or eval_result.get("status") == "error" or "evaluation" not in eval_result:
                error_count += 1
                continue

            correctness = eval_result["evaluation"]["correctness"]
            num_parts = len(correctness)
            score = 0.0
            if num_parts == 1:
                score = 1.0 if correctness[0] else 0.0
            elif num_parts in SUB_QUESTION_WEIGHTS:
                weights = SUB_QUESTION_WEIGHTS[num_parts]
                score = sum(w for c, w in zip(correctness, weights) if c)
            else:
                # Skip items with more answer parts than defined weights
                too_many_parts_count += 1
                continue

            # --- Update Statistics ---

            # 1. Overall Stats
            stats['overall']['count'] += 1
            stats['overall']['total_score'] += score
            if all(correctness):
                stats['overall']['completely_correct_count'] += 1

            # 2. Image Presence Stats
            q_images = [p for p in ori_data.get("question_interleave", []) if p['type'] == 'image']
            image_presence_key = "Has Image" if len(q_images) > 0 else "No Image"
            self._update_stats(stats['by_question_image_count'], image_presence_key, score)

            # 3. Knowledge Stats
            knowledge_key = ori_data.get("knowledge", "Unknown")
            self._update_stats(stats['by_knowledge'], knowledge_key, score)

        # --- Final Report Generation ---
        def calculate_accuracy(data_dict):
            report = {}
            for key, value in sorted(data_dict.items()):
                count = value.get('count', 0)
                total_score = value.get('total_score', 0.0)
                accuracy = round((total_score / count) * 100, 1) if count > 0 else 0.0
                report[key] = {'count': count, 'accuracy': accuracy}
            return report

        total_valid = stats['overall']['count']
        
        # Build the simplified summary report
        summary_report = {
            'overall_summary': {
                'total_evaluated': len(all_items),
                'valid_evaluations': total_valid,
                'processing_errors': error_count,
                'too_many_parts_skipped': too_many_parts_count,
                'weighted_accuracy': round((stats['overall']['total_score'] / total_valid) * 100, 1) if total_valid > 0 else 0.0,
                'complete_accuracy': round((stats['overall']['completely_correct_count'] / total_valid) * 100, 1) if total_valid > 0 else 0.0,
            },
            'knowledge_summary': calculate_accuracy(stats['by_knowledge']),
            'accuracy_by_question_image_count': calculate_accuracy(stats['by_question_image_count']),
        }
        
        # --- Save report files ---
        summary_dir = os.path.dirname(self.args.inference_dir.rstrip('/'))
        inference_name = os.path.basename(self.args.inference_dir.rstrip('/'))
        summary_path = os.path.join(summary_dir, f"{inference_name}--{self.model_name}--evaluation_summary_valid{total_valid}.json")
        
        save_json(summary_report, summary_path)

        all_results_path = os.path.join(summary_dir, f"{inference_name}--{self.model_name}--evaluation_all_results.jsonl")
        with open(all_results_path, "w", encoding="utf-8") as f:
            for res in all_results_data:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")

        print("\n--- Evaluation Summary ---")
        print(json.dumps(summary_report, indent=2))
        print(f"\nSummary saved to: {summary_path}")
        print(f"Detailed results saved to: {all_results_path}")

    def run(self):
        """Main execution function."""
        all_subdirs = [d for d in glob.glob(os.path.join(self.args.inference_dir, "*")) if os.path.isdir(d)]
        items_to_process = self.find_items_to_process()
        
        if not items_to_process:
            if not all_subdirs:
                print(f"Inference directory '{self.args.inference_dir}' is empty or contains no subdirectories.")
                return
            print("No new items to evaluate.")

            summary_dir = os.path.dirname(self.args.inference_dir.rstrip('/'))
            inference_name = os.path.basename(self.args.inference_dir.rstrip('/'))
            summary_path_pattern = os.path.join(summary_dir, f"{inference_name}--{self.model_name}--evaluation_summary*.json")
            if not glob.glob(summary_path_pattern):
                print("Generating missing summary file...")
                all_item_dicts = [{
                    "id": os.path.basename(d), 
                    "output_path": os.path.join(d, f"{self.model_name}--evaluation_result.json"),
                    "ori_json_path": os.path.join(d, "ori.json"), 
                } for d in all_subdirs if os.path.exists(os.path.join(d, f"{self.model_name}--evaluation_result.json"))
                ]
                self.summarize_results(all_item_dicts)
            else:
                print(f"Summary file already exists: {glob.glob(summary_path_pattern)[0]}")
            return

        print(f"Found {len(items_to_process)} items to evaluate.")

        with ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
            list(tqdm(executor.map(self.process_single_item, items_to_process), total=len(items_to_process), desc="Evaluating Answers"))
        
        print("\nEvaluation complete. Generating final summary...")
        all_item_dicts_after_run = [{
            "id": os.path.basename(d), 
            "output_path": os.path.join(d, f"{self.model_name}--evaluation_result.json"),
            "ori_json_path": os.path.join(d, "ori.json"), 
        } for d in all_subdirs if os.path.exists(os.path.join(d, f"{self.model_name}--evaluation_result.json"))
        ]
        self.summarize_results(all_item_dicts_after_run)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch evaluate model answers for math problems using an API.")
    
    parser.add_argument("--inference_dir", type=str, required=True, help="Path to the directory of a specific inference run.")
    parser.add_argument("--prompt_file", type=str, default="mathcanvas_evaluate_prompt.txt", help="Path to the prompt template file.")
    
    # API method is now fixed to OpenAI
    parser.add_argument("--model", type=str, required=True, help="The model name for the API call (e.g., gpt-4.1-2025-04-14).")
    parser.add_argument("--api_key", type=str, required=True, help="Your API key.")
    parser.add_argument("--base_url", type=str, required=True, help="Base URL for API services.")
    
    parser.add_argument("--max_workers", type=int, default=16, help="Max parallel threads for API calls.")
    parser.add_argument("--force_rerun", action='store_true', help="Force re-evaluation of all items.")
    
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature for the API model.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for the API response.")
    parser.add_argument("--reasoning_effort", type=str, default='none', choices=['none', 'minimal', 'low', 'medium', 'high'], help="Reasoning effort level for OpenAI API. Defaults to none.")
    
    args = parser.parse_args()

    evaluator = MathAnswerEvaluator(args)
    evaluator.run()