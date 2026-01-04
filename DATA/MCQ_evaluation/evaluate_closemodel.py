import json
import os
import argparse
from typing import List, Dict, Any
import openai
from tqdm import tqdm
# import anthropic
# import google.generativeai as genai
import requests
import time
import re
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import random
import asyncio
import aiohttp

class GeneratorBenchmark:
    # JSON Schema for structured output
    JSON_SCHEMA = {
        "type": "OBJECT",
        "properties": {
            "choice": {"type": "STRING", "enum": ["A","B","C","D"]},
            "reason": {"type": "STRING"}
        },
        "required": ["choice"]
    }

    PROMPT_TEMPLATE_SINGLE = """Question: {q}
Options:
{opts}
{passage_block}

Answer only the letter: A, B, C, or D""".strip()

    PROMPT_TEMPLATE_MULTI = """You are given 5 passages (some may be irrelevant). You must select ONLY ONE passage that is most relevant to answer the question. Write the passage used: Passage: <single number between 1 and 5> (only one number, no commas or multiple numbers), then provide your answer.

{passages_block}

Question: {q}
Options:
{opts}

Answer only the letter: A, B, C, or D""".strip()
    
    def __init__(self, test_set_path: str, model_name: str = None):
        """
        Initialize the generator benchmark
        
        Args:
            test_set_path: Path to the test set (base/golden/mix)
            model_name: Name of the model (for golden lookup optimization)
        """
        # Load .env file from DATA directory
        env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
        load_dotenv(dotenv_path=env_path)
        
        self.test_set = self._load_test_set(test_set_path)
        self.setting = os.path.basename(test_set_path).split('_')[0]  # base/golden/mix
        self.model_name = model_name
        
        # Load passages_by_score data for mix setting (to construct 5 passages)
        self.passages_data = {}
        if self.setting == "mix":
            self._load_passages_data()
        
        # API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        # self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        

        
        # Thread safety
        self.print_lock = Lock()
        self.results_lock = Lock()
        
        # Set random seed for reproducibility
        # random.seed(42)
        

    
    async def __aenter__(self):
        """Initialize async context"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async context"""
        pass
    
    def _load_test_set(self, test_set_path: str) -> List[Dict]:
        """Load the test set"""
        with open(test_set_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _load_passages_data(self):
        """Load passages_by_score data from DATA/DATA/data_prepare for mix setting"""
        try:
            # Get the project root directory (3 levels up from this script)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(os.path.dirname(script_dir))
            data_prepare_dir = os.path.join(base_dir, "DATA", "DATA", "data_prepare")
            
            if not os.path.exists(data_prepare_dir):
                print(f"Warning: data_prepare directory not found at {data_prepare_dir}")
                return
            
            # Load all *_by_score.json files
            import glob
            json_files = glob.glob(os.path.join(data_prepare_dir, "*_by_score.json"))
            
            print(f"Loading passages data from {len(json_files)} files...")
            
            for json_file in json_files:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                    for item in data:
                        user_query = item.get('user_query', '')
                        if user_query:
                            passages_by_score = item.get('passages_by_score', {})
                            if passages_by_score:
                                self.passages_data[user_query] = passages_by_score
            
            print(f"Loaded passages data for {len(self.passages_data)} queries")
            
        except Exception as e:
            print(f"Error loading passages data: {e}")
            import traceback
            traceback.print_exc()
    
    def _construct_five_passages(self, original_query: str) -> List[str]:
        """Construct 5 passages for mix setting: 1 golden (score=3) + 4 distractors (one from each score 0,1,2,3)"""
        if original_query not in self.passages_data:
            print(f"Warning: No passages data found for query: {original_query[:50]}...")
            return None
        
        passages_by_score = self.passages_data[original_query]
        
        # Get golden passage (score=3)
        golden_passages = passages_by_score.get('3', [])
        if not golden_passages:
            print(f"Warning: No golden passages (score=3) found for query: {original_query[:50]}...")
            return None
        
        golden_passage = golden_passages[0]  # Use first golden passage
        
        # Get one distractor from each score (0, 1, 2)
        distractors = []
        for score in ['0', '1', '2']:
            score_passages = passages_by_score.get(score, [])
            if score_passages:
                distractors.append(random.choice(score_passages))
            else:
                # If no passage for this score, use another from available scores
                for alt_score in ['0', '1', '2']:
                    if alt_score != score and alt_score in passages_by_score:
                        alt_passages = passages_by_score[alt_score]
                        if alt_passages:
                            distractors.append(random.choice(alt_passages))
                            break
        
        # Add one more distractor if we don't have 4 yet (can be from any low score)
        if len(distractors) < 4:
            all_low_passages = []
            for score in ['0', '1', '2']:
                all_low_passages.extend(passages_by_score.get(score, []))
            if all_low_passages:
                while len(distractors) < 4 and all_low_passages:
                    distractors.append(random.choice(all_low_passages))
                    all_low_passages.remove(distractors[-1])
        
        # Combine: 1 golden + 4 distractors = 5 passages total
        five_passages = [golden_passage] + distractors[:4]
        
        # Randomly shuffle the order (so golden is not always first)
        random.shuffle(five_passages)
        
        return five_passages
    
    async def generate_answer(self, query: str, context: List[str], model: str, options: List[str], five_passages: List[str] = None) -> str:
        """Generate answer for MCQ using structured JSON output"""
        # Build options string - remove original prefixes and re-number sequentially  
        clean_options = [opt.split('. ', 1)[1] if '. ' in opt else opt for opt in options]
        options_str = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(clean_options)])
        
        # Build prompt based on setting
        if five_passages and len(five_passages) == 5:
            # Mix setting: 5 passages with selection instruction
            passages_block = "\n\n".join([f"Passage {i+1}: {passage}" for i, passage in enumerate(five_passages)])
            prompt = self.PROMPT_TEMPLATE_MULTI.format(q=query, opts=options_str, passages_block=passages_block)
        elif context:
            # Golden setting: single passage
            passage_block = "Passage:\n" + "\n".join(context)
            prompt = self.PROMPT_TEMPLATE_SINGLE.format(q=query, opts=options_str, passage_block=passage_block)
        else:
            # Base setting: no passage
            prompt = self.PROMPT_TEMPLATE_SINGLE.format(q=query, opts=options_str, passage_block="")
        
        # Call the appropriate model
        return await self._call_model(model, prompt)
    
    async def _call_model(self, model_name: str, prompt: str) -> str:
        """Call the appropriate model based on model name.
        
        Args:
            model_name: Name of the model to use
            prompt: The prompt to send to the model
            
        Returns:
            str: Model's response
        """
        if model_name in ["gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gpt-5.2"]:  # Add gpt-5.2
            return await self._call_openai(model_name, prompt)
        elif model_name in ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro", "gemini-3-pro"]:  # Add gemini-3-pro
            return await self._call_google(model_name, prompt)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    async def _call_openai(self, model: str, prompt: str) -> str:
        """Call OpenAI API asynchronously."""
        client = openai.AsyncOpenAI(api_key=self.openai_api_key)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers multiple choice questions based on the given context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Very low temperature for deterministic answers
                max_tokens=1000   # Add max_tokens parameter
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling OpenAI API for model {model}: {str(e)}")
            raise
    
    # def _call_anthropic(self, prompt: str) -> str:
    #     client = anthropic.Anthropic(api_key=self.anthropic_api_key)
    #     response = client.messages.create(
    #         model="claude-2",
    #         messages=[{"role": "user", "content": prompt}]
    #     )
    #     return response.content[0].text
    
    async def _call_google(self, model_name: str, prompt: str) -> str:
        """Call Google Gemini API using standard REST API."""
        max_retries = 5
        base = 1.0  # Start with 1 second base delay
        
        for attempt in range(max_retries):
            try:
                # use standard Generative Language API
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
                headers = {"Content-Type": "application/json"}
                params = {"key": self.google_api_key}
                
                data = {
                    "systemInstruction": {
                        "parts": [{"text": "You are a helpful assistant that answers multiple choice questions based on the given context."}]
                    },
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {
                        "temperature": 0.0,
                        "maxOutputTokens": 1000,  # enough space for Gemini 2.5 Pro's thinking and output, but do not save detailed response
                    },
                    "safetySettings": [
                        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"}
                    ]
                }
                
                # check API Key
                if not self.google_api_key:
                    raise Exception("Google API Key not found in environment variables")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=data, headers=headers, params=params) as response:
                        if response.status == 200:
                            result = await response.json()
                            # Handle different response formats
                            if "candidates" in result and result["candidates"]:
                                candidate = result["candidates"][0]
                                finish_reason = candidate.get("finishReason", "")
                                
                                # Handle MAX_TOKENS case - return empty response
                                if finish_reason == "MAX_TOKENS":
                                    return "Unable to complete response due to token limit"
                                
                                if "content" in candidate:
                                    content = candidate["content"]
                                    if "parts" in content and content["parts"]:
                                        # normal case: has parts array
                                        return content["parts"][0]["text"]
                                    elif "text" in content:
                                        # alternative: directly text field
                                        return content["text"]
                                    elif content.get("role") == "model" and not content.get("parts"):
                                        # special case for Gemini 2.5 Pro: content only has role, no actual content
                                        # this usually means the model did not generate any text, possibly a prompt problem
                                        return "Unable to generate response - empty model output"
                                elif "text" in candidate:
                                    return candidate["text"]
                                
                            # if no return, throw exception for retry fallback
                            raise Exception(f"Unexpected response format: {result}")
                        elif response.status in (429, 500, 502, 503, 504):
                            # explicitly throw retryable error
                            text = await response.text()
                            raise RuntimeError(f"RETRYABLE {response.status}: {text}")
                        else:
                            # not retryable
                            text = await response.text()
                            raise Exception(f"API call failed {response.status}: {text}")
                
            except Exception as e:
                msg = str(e).lower()
                retryable = isinstance(e, RuntimeError) or "quota" in msg or "rate" in msg or "limit" in msg
                if retryable and attempt < max_retries - 1:
                    # For 429 errors, use longer wait time
                    if "429" in str(e) or "quota" in str(e).lower():
                        wait = 60 + random.uniform(0, 30)  # Wait 60-90 seconds
                    else:
                        wait = base * (2 ** attempt) + random.uniform(0, 0.5)  # Other errors normal backoff
                    print(f"[Retry {attempt+1}/{max_retries}] {model_name}: {e} -> sleep {wait:.2f}s")
                    await asyncio.sleep(wait)
                    continue
                print(f"Error calling Google Gemini API for model {model_name}: {e}")
                raise
    

    
    def run_benchmark(self, model: str) -> Dict:
        """Run benchmark for a specific model"""
        results = {
            "predictions": [],
            "references": [],
            "correct": 0,
            "total": 0
        }
        
        for case in tqdm(self.test_set, desc=f"Running {model} on {self.setting}"):
            # Get question and options
            question = case["multiple_choice"]["gpt40"]["content"]["question"]
            options = case["multiple_choice"]["gpt40"]["content"]["options"]
            correct_answer = case["multiple_choice"]["gpt40"]["content"]["correct_option"]
            
            # Get context based on setting
            if self.setting == "base":
                context = []
            else:
                context = [case["passage"]] if "passage" in case else []
            
            # Generate answer
            answer = self.generate_answer(question, context, model, options)
            
            # Store results
            results["predictions"].append(answer)
            results["references"].append(correct_answer)
            results["total"] += 1
            
            # Check if correct
            if answer.strip().upper() == correct_answer.strip().upper():
                results["correct"] += 1
        
        # Calculate accuracy
        results["accuracy"] = results["correct"] / results["total"]
        
        return results

#     def _generate_prompt(self, item: Dict[str, Any]) -> str:
#         """Generate prompt for the model based on test set item."""
#         # Get question and options
#         if "multiple_choice" in item and "gpt40" in item["multiple_choice"]:
#             content = item["multiple_choice"]["gpt40"]["content"]
#             question = content["question"]
#             options = content["options"]
#         else:
#             raise ValueError("Invalid test set format: missing multiple_choice or gpt40 content")

#         # Get retrieved documents if available
#         retrieved_docs = item.get("retrieved_documents", [])
        
#         # Build prompt
#         prompt = f"""Please answer the following multiple choice question. Choose the most appropriate answer from the given options.

# Question: {question}

# Options:
# """
#         # Add options
#         for i, opt in enumerate(options):
#             prompt += f"{chr(65 + i)}. {opt.split('. ')[1]}\n"

#         if retrieved_docs:
#             prompt += "\nRelevant information:\n"
#             for i, doc in enumerate(retrieved_docs, 1):
#                 prompt += f"{i}. {doc}\n"
        
#         prompt += "\nPlease provide your answer in the following format:\nAnswer: [A/B/C/D]\nExplanation: [Your reasoning]"
        
#         return prompt

    def _parse_model_response(self, response: str) -> Dict[str, str]:
        """Parse model's response to extract answer and explanation."""
        # Try to find answer and explanation in the response
        answer = None
        explanation = None
        
        # Look for "Answer: X" pattern
        answer_match = re.search(r"Answer:\s*([A-D])", response, re.IGNORECASE)
        if answer_match:
            answer = answer_match.group(1)
        
        # Look for "Explanation:" or "Reasoning:" followed by text
        explanation_match = re.search(r"(?:Explanation|Reasoning):\s*(.*?)(?:\n|$)", response, re.DOTALL | re.IGNORECASE)
        if explanation_match:
            explanation = explanation_match.group(1).strip()
        
        return {
            "answer": answer,
            "explanation": explanation
        }

    async def process_single_item(self, item: Dict[str, Any], model_name: str, item_index: int, total: int) -> Dict[str, Any]:
        """Process a single test item."""
        try:
            # Get question and options
            question = item["multiple_choice"]["gpt40"]["content"]["question"]
            options = item["multiple_choice"]["gpt40"]["content"]["options"]
            correct_answer = item["multiple_choice"]["gpt40"]["content"]["correct_option"]
            
            # Get context and five_passages based on setting
            context = []
            passage = ""
            five_passages = None
            
            if self.setting == "base":
                context = []
                passage = ""
            elif self.setting == "golden":
                context = [item["passage"]] if "passage" in item else []
                passage = item.get("passage", "")
                if not context:
                    return None  # Skip items without passage for golden setting
            elif self.setting == "mix":
                # For mix setting, construct 5 passages
                original_query = item.get("original_query", "")
                if original_query:
                    five_passages = self._construct_five_passages(original_query)
                passage = item.get("passage", "")  # Save original passage for reference
            
            # Generate answer
            raw_answer = await self.generate_answer(question, context, model_name, options, five_passages)
            
            # Parse the answer and explanation
            answer, explanation = self._parse_response(raw_answer)
            if not answer:
                # If parsing fails, take the first A/B/C/D found in the response
                for char in raw_answer:
                    if char.upper() in ['A', 'B', 'C', 'D']:
                        answer = char.upper()
                        break
                if not answer:  # If still no answer found, randomly choose one
                    answer = random.choice(['A', 'B', 'C', 'D'])
            
            # Create result (removed model_explanation to save tokens)
            result = {
                "question": question,
                "options": options,
                "correct_answer": correct_answer,
                "model_answer": answer,
                "is_correct": answer == correct_answer
            }
            
            # ensure passage field is correctly passed to the result
            if passage:
                result['passage'] = passage
            
            # add original_query field for classification analysis
            if 'original_query' in item:
                result['original_query'] = item['original_query']
            
            return result
            
        except Exception as e:
            print(f"[ERROR] item {item_index}/{total}: {e}")
            if 'raw_answer' in locals():
                print(f"Raw answer: {raw_answer}")
            else:
                print("Raw answer: N/A")
            return None

    def _parse_response(self, response: str) -> tuple[str, str]:
        """Parse the model's response to extract answer letter only.
        
        Args:
            response: The model's full response text
            
        Returns:
            tuple: (answer_letter, response)
        """
        resp = response.strip()
        
        # 1) find the first independent A/B/C/D (most common case)
        m = re.search(r'\b([A-D])\b', resp, flags=re.IGNORECASE)
        if m:
            return m.group(1).upper(), resp
        
        # 2) if not found, return empty answer
        return "", resp

    async def process_test_set(self, test_set: List[Dict[str, Any]], model_name: str, num_tests: int = None, concurrency: int = 500) -> List[Dict[str, Any]]:
        """Process test set with specified model using asyncio.
        
        Args:
            test_set: List of test cases
            model_name: Name of the model to use
            num_tests: Number of test cases to process. If None, process all.
            concurrency: Maximum number of concurrent requests
        """
        results = []
        total = len(test_set)
        
        # If num_tests is specified, randomly select that many cases
        if num_tests is not None:
            test_set = random.sample(test_set, min(num_tests, total))
            total = len(test_set)
            print(f"\nRandomly selected {total} items for testing...")
        else:
            print(f"\nProcessing all {total} items...")
        
        # Create semaphore to limit concurrency
        sem = asyncio.Semaphore(concurrency)
        
        async def process_with_semaphore(item, index):
            async with sem:
                return await self.process_single_item(item, model_name, index, total)
        
        # Create tasks
        tasks = [
            process_with_semaphore(item, i+1)
            for i, item in enumerate(test_set)
        ]
        
        # Process all tasks
        for result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=f"Processing {self.setting} test set"):
            result = await result
            if result is not None:
                results.append(result)
        
        # Calculate statistics
        total_answers = len(results)
        if total_answers == 0:
            print(f"\nNo valid results for {self.setting} setting")
            return results
            
        
        print(f"\nConfidence Statistics for {self.setting}:")
        print(f"Total processed: {total_answers}/{total} ({total_answers/total*100:.1f}%)")
        
        return results

def load_test_set(file_path: str) -> List[Dict[str, Any]]:
    """Load test set from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_results(results: List[Dict[str, Any]], output_path: str):
    """Save results to JSON file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

def print_test_set_statistics(test_sets: List[str]):
    """Print statistics for all test sets."""
    print("\nTest set statistics:")
    for test_set in test_sets:
        with open(test_set, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Get all questions
            questions = [item["multiple_choice"]["gpt40"]["content"]["question"] for item in data]
            # Count unique questions
            unique_questions = set(questions)
            print(f"\n{os.path.basename(test_set)}:")
            print(f"Total items: {len(data)}")
            print(f"Unique questions: {len(unique_questions)}")
            # Print some example questions
            print("Example questions:")
            for q in list(unique_questions)[:3]:
                print(f"- {q}")

def main():
    # Specify the model to use
    # Options: "gpt-3.5-turbo", "gpt-4o-mini", "gpt-4o", "gemini-2.5-pro", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"
    model_name = "gemini-1.5-pro"  # Change this to the model you want to use
    
    # Set number of test cases to process (None for all)
    num_tests = None  # Process ALL test cases (2000 per setting)
    
    # Set concurrency limit - optimal concurrency for Gemini 1.5 Pro
    # Based on testing, 500 concurrent seems optimal
    concurrency = 500 if model_name.startswith("gemini") else 3000  # Optimal 500 concurrent
    
    # Set paths (relative to project root)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(os.path.dirname(script_dir))
    
    test_sets = [
        os.path.join(base_dir, "DATA", "final_mcq", "base_2000.json"),
        os.path.join(base_dir, "DATA", "final_mcq", "golden_2000.json"),
        os.path.join(base_dir, "DATA", "final_mcq", "mix_2000.json")
    ]
    
    # Create output directory
    output_dir = os.path.join(base_dir, "DATA", "local_MCQ", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Print statistics for all test sets
    print_test_set_statistics(test_sets)
    
    print(f"\n{'='*50}")
    print(f"Starting benchmark with {num_tests if num_tests else 'all'} test cases")
    print(f"{'='*50}\n")
    
    # Run benchmarks
    async def run_benchmarks():
        for test_set in test_sets:
            setting = os.path.basename(test_set).split('_')[0]  # base/golden/mix
            output_path = os.path.join(output_dir, f"{setting}_test.json")
            
            # Check if output file already exists
            if os.path.exists(output_path):
                print(f"\nOutput file for {setting} setting already exists at: {output_path}")
                print("Skipping this setting...")
                continue
            
            print(f"\nTesting {model_name} on {setting} setting...")
            
            async with GeneratorBenchmark(test_set, model_name) as benchmark:
                results = await benchmark.process_test_set(benchmark.test_set, model_name, num_tests, concurrency)
                
                # Save results for this setting
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                
                print(f"\nResults for {setting} saved to: {output_path}")
    
    # Run the async main function
    asyncio.run(run_benchmarks())

if __name__ == "__main__":
    main()