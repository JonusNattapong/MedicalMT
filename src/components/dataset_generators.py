# -*- coding: utf-8 -*-
"""
Dataset generation components for MedMT
"""
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import random
import logging
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Dict, List, Any
from .display_utils import LoadingSpinner, ProgressBar
from .text_utils import extract_thai_translation, _create_diverse_prompt, _get_dynamic_parameters
from .metadata_utils import add_metadata

def _process_dialogue_sample(args_tuple):
    """Helper function to process a single sample for generate_deepseek_medical_dialogue"""
    i, topic, sample_data, client, prompt_template, log_error_func = args_tuple
    
    prompt = _create_diverse_prompt(topic, sample_data, i, prompt_template)
    dynamic_params = _get_dynamic_parameters(i)
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical translator specialized in Chinese to Thai translation with expertise in healthcare terminology. "
                        "Your task is to translate the Chinese source text to Thai with precision, considering medical context. "
                        "Instructions for translation:"
                        "\n1. Translate accurately and completely, preserving all medical terms and concepts."
                        "\n2. Maintain the questioning tone where appropriate - if source ends with '吗', add '?' in Thai."
                        "\n3. Preserve medical terminology with proper Thai medical equivalents."
                        "\n4. Consider the symptom and context to ensure translation is contextually appropriate."
                        "\n5. Use natural Thai phrasing that would be used by Thai medical professionals."
                        "\n6. Maintain formal medical register when appropriate."
                        "\n7. Respond with ONLY the Thai translation, no explanations or additional text."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            temperature=dynamic_params["temperature"],
            presence_penalty=dynamic_params["presence_penalty"],
            frequency_penalty=dynamic_params["frequency_penalty"],
            top_p=dynamic_params["top_p"]
        )
        full_response = response.choices[0].message.content.strip()
        target = extract_thai_translation(full_response)
        
        if not target:
            target = sample_data.get("target", "Error: No translation extracted")
            msg = f"Failed to extract Thai translation. Response: '{full_response}'. Prompt: '{prompt[:200]}...'"
            print(f"[Warning] {msg} at sample index {i}, using sample target or error message.")
            log_error_func("TranslationExtractionFailure", i, msg)
            
        return {"context": sample_data["context"], "source": sample_data["source"], "target": target, "original_index": i}
    except Exception as e:
        target = sample_data.get("target", f"Error: API call failed - {type(e).__name__}")
        msg = f"DeepSeek API error: {e}. Prompt: {prompt[:200]}..."
        print(f"[ERROR] {msg} at sample index {i}")
        log_error_func("DeepSeekAPIError", i, msg)
        return {"context": sample_data["context"], "source": sample_data["source"], "target": target, "original_index": i, "error": str(e)}

def generate_deepseek_medical_dialogue(n_samples: int = 100, seed: int = 42, max_workers: int = 5) -> pd.DataFrame:
    """Generate dialogue dataset using DeepSeek AI with parallel processing"""
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in .env file")
    
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    
    random.seed(seed)
    print(f"[INFO] Preparing prompt combinations for {n_samples} dialogue samples...")
    
    # Import templates
    try:
        from src.dataset_templates import MEDICAL_TOPICS, DIALOGUE_SAMPLES, DIALOGUE_PROMPT_TEMPLATE
    except ImportError:
        print("[ERROR] Could not import dialogue templates")
        return pd.DataFrame()
    
    all_prompt_combinations = []
    for topic in MEDICAL_TOPICS:
        for sample in DIALOGUE_SAMPLES:
            all_prompt_combinations.append((topic, sample))
    
    if not all_prompt_combinations:
        print("[ERROR] No dialogue samples or medical topics found.")
        return pd.DataFrame()

    selected_combinations = []
    if n_samples <= len(all_prompt_combinations):
        selected_combinations = random.sample(all_prompt_combinations, n_samples)
    else:
        selected_combinations.extend(all_prompt_combinations)
        remaining_samples = n_samples - len(all_prompt_combinations)
        for i in range(remaining_samples):
            selected_combinations.append(all_prompt_combinations[i % len(all_prompt_combinations)])
    
    tasks_args = []
    for i, (topic, sample) in enumerate(selected_combinations):
        tasks_args.append((i, topic, sample, client, DIALOGUE_PROMPT_TEMPLATE, logging.error))
    
    print(f"[INFO] Created {len(tasks_args)} tasks for dialogue generation.")
    
    data = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(_process_dialogue_sample, tasks_args)
        progress = ProgressBar(len(tasks_args), prefix='Generating Dialogue Samples:', suffix='Complete')
        for result in results_iterator:
            data.append(result)
            progress.increment()

    processed_data = []
    for item in data:
        item_copy = item.copy()
        item_copy.pop("original_index", None)
        item_copy.pop("error", None)
        processed_data.append(item_copy)

    df = pd.DataFrame(processed_data)
    if not df.empty:
        df = add_metadata(df)
    return df

def generate_qa_dataset(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate question-answer pairs dataset using DeepSeek AI"""
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in .env file")
    
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    data = []
    random.seed(seed)
    
    # Import QA_SAMPLES and QA_PROMPT_TEMPLATE
    try:
        from src.dataset_templates import QA_SAMPLES, QA_PROMPT_TEMPLATE
    except ImportError:
        print("[ERROR] Could not import QA templates")
        return pd.DataFrame()

    if not QA_SAMPLES:
        print("[ERROR] No QA_SAMPLES found in dataset_templates.py.")
        return pd.DataFrame()

    # Initialize progress bar
    progress = ProgressBar(n_samples, prefix='Generating QA Samples:', suffix='Complete')
    
    for i in range(n_samples):
        sample = QA_SAMPLES[i % len(QA_SAMPLES)]
        
        # Create diverse questions
        question_templates = [
            "这可能是什么疾病？需要做什么检查？",
            "根据症状，可能是什么病？应该进行哪些检查？",
            "这些症状表明什么疾病？建议做什么检测？",
            "从医学角度看，可能是什么疾病？需要哪些检查？",
            "诊断可能是什么？应该进行哪些医学检查？",
            sample["question"]
        ]
        
        question_index = hash(str(i) + "qa_q_salt") % len(question_templates)
        diverse_question = question_templates[question_index]
        
        prompt = QA_PROMPT_TEMPLATE.format(
            topic=sample["topic"],
            context=sample["context"],
            question=diverse_question,
            answer=sample["answer"]
        )
        
        dynamic_params = _get_dynamic_parameters(i, base_temp=0.65)
        
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are a medical translator specialized in Chinese to Thai translation. "
                            "Your task is to translate both questions and answers from Chinese to Thai in the medical domain."
                            "\n1. Translate both the question and answer accurately, using appropriate Thai medical terminology."
                            "\n2. Preserve all medical concepts and reasoning."
                            "\n3. Keep medical terms precise and maintain natural Thai language flow."
                            "\n4. Format your response as:\n"
                            "Q: [Thai question]\n"
                            "A: [Thai answer]"
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=dynamic_params["temperature"],
                presence_penalty=dynamic_params["presence_penalty"],
                frequency_penalty=dynamic_params["frequency_penalty"],
                top_p=dynamic_params["top_p"]
            )
            
            full_response = response.choices[0].message.content.strip()
            
            q_match = re.search(r'Q:\s*(.+)', full_response, re.DOTALL)
            a_match = re.search(r'A:\s*(.+)', full_response, re.DOTALL)
            
            question_th = q_match.group(1).strip() if q_match else sample.get("q_th", "Error: Q not extracted")
            answer_th = a_match.group(1).strip() if a_match else sample.get("a_th", "Error: A not extracted")
            
            if "Error:" in question_th or "Error:" in answer_th:
                logging.error(f"QAParsingFailure | Sample: {i} | Failed to parse Q/A. Response: '{full_response}'. Prompt: '{prompt[:200]}...'")

        except Exception as e:
            question_th = sample.get("q_th", f"Error: API call failed - {type(e).__name__}")
            answer_th = sample.get("a_th", f"Error: API call failed - {type(e).__name__}")
            logging.error(f"DeepSeekAPIError | Sample: {i} | QA Gen Error: {e}. Prompt: {prompt[:200]}...")
        
        data.append({
            "context": sample["context"],
            "question_zh": diverse_question,
            "answer_zh": sample["answer"],
            "question_th": question_th,
            "answer_th": answer_th
        })
        progress.increment()
            
    df = pd.DataFrame(data)
    if not df.empty:
        df = add_metadata(df)
    return df