# -*- coding: utf-8 -*-
"""
Multi-turn dialogue generators for MedMT - Generate conversations with multiple turns
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
from .response_cache import cached_api_call

MULTI_TURN_PROMPT_TEMPLATE = """Thai-Chinese Medical Translation Task - Multi-turn Dialogue

ฉันต้องการให้คุณสร้างบทสนทนาหลายรอบ (3-4 รอบ) ระหว่างแพทย์และผู้ป่วยในหัวข้อทางการแพทย์ต่อไปนี้:
--------------------------------
หัวข้อ (จีน): {topic_zh}
หัวข้อ (ไทย): {topic_th}
คำอธิบาย: {topic_desc}

บริบท: {context}

คำแนะนำ:
1. สร้างบทสนทนาที่มีความสมจริงระหว่างแพทย์และผู้ป่วย จำลองการปรึกษาทางการแพทย์ที่มีหลายคำถามและคำตอบ
2. เริ่มต้นด้วยผู้ป่วยบรรยายอาการ ตามด้วยแพทย์ถามคำถามเพิ่มเติม 
3. ให้คุณสร้างทั้งข้อความภาษาจีน และคำแปลภาษาไทยที่เหมาะสม ใช้คำศัพท์ทางการแพทย์ที่ถูกต้อง
4. ใช้รูปแบบดังนี้สำหรับแต่ละรอบการสนทนา:

รอบที่ 1:
- ผู้ป่วย (จีน): [ข้อความจีน]
- ผู้ป่วย (ไทย): [คำแปลไทย]
- แพทย์ (จีน): [ข้อความจีน]
- แพทย์ (ไทย): [คำแปลไทย]

รอบที่ 2:
- ผู้ป่วย (จีน): [ข้อความจีน]
- ผู้ป่วย (ไทย): [คำแปลไทย]
- แพทย์ (จีน): [ข้อความจีน]
- แพทย์ (ไทย): [คำแปลไทย]
...และต่อไปเรื่อยๆ

โปรดอย่าละเว้นคำแปลไทยในรอบใดๆ
"""

def _process_multiturn_sample(args_tuple):
    """Process a single multi-turn dialogue sample"""
    i, topic, sample_data, client, prompt_template, log_error_func = args_tuple
    
    prompt = prompt_template.format(
        topic_zh=topic["zh"],
        topic_th=topic["th"],
        topic_desc=topic["desc"],
        context=sample_data["context"]
    )
    
    dynamic_params = _get_dynamic_parameters(i, base_temp=0.75)
    
    try:
        response_dict = cached_api_call(
            client,
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a medical dialogue generator specializing in Chinese-Thai bilingual content. "
                        "Create realistic multi-turn conversations between doctors and patients with accurate medical terminology. "
                        "Follow the format exactly as requested, providing both Chinese and Thai versions."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=dynamic_params["temperature"],
            presence_penalty=dynamic_params["presence_penalty"],
            frequency_penalty=dynamic_params["frequency_penalty"],
            top_p=dynamic_params["top_p"]
        )
        
        # Extract response from the cached response dict
        full_response = response_dict["choices"][0]["message"]["content"].strip()
        
        # Process the full_response to extract multi-turn conversations
        conversations = []
        current_turn = {"turn": 0}
        
        # Match each turn section
        turn_matches = re.findall(r'รอบที่\s*(\d+):(.*?)(?=รอบที่\s*\d+:|$)', full_response, re.DOTALL)
        
        if turn_matches:
            for turn_num, turn_content in turn_matches:
                turn_data = {"turn": int(turn_num)}
                
                # Extract patient's text
                patient_zh = re.search(r'ผู้ป่วย\s*\(จีน\):\s*(.+?)(?=-|\n|ผู้ป่วย\s*\(ไทย\))', turn_content, re.DOTALL)
                patient_th = re.search(r'ผู้ป่วย\s*\(ไทย\):\s*(.+?)(?=-|\n|แพทย์\s*\(จีน\))', turn_content, re.DOTALL)
                
                # Extract doctor's text
                doctor_zh = re.search(r'แพทย์\s*\(จีน\):\s*(.+?)(?=-|\n|แพทย์\s*\(ไทย\))', turn_content, re.DOTALL)
                doctor_th = re.search(r'แพทย์\s*\(ไทย\):\s*(.+?)(?=-|\n|$)', turn_content, re.DOTALL)
                
                # Add to turn data if found
                if patient_zh and patient_th and doctor_zh and doctor_th:
                    turn_data["patient_zh"] = patient_zh.group(1).strip()
                    turn_data["patient_th"] = patient_th.group(1).strip()
                    turn_data["doctor_zh"] = doctor_zh.group(1).strip()
                    turn_data["doctor_th"] = doctor_th.group(1).strip()
                    
                    conversations.append(turn_data)
        
        if not conversations:
            log_error_func("MultiTurnExtractionFailure", i, f"Failed to extract multi-turn conversation. Response: '{full_response[:200]}...'")
            return {
                "context": sample_data["context"], 
                "topic_zh": topic["zh"],
                "topic_th": topic["th"],
                "turns": [],
                "original_index": i,
                "success": False
            }
            
        return {
            "context": sample_data["context"], 
            "topic_zh": topic["zh"],
            "topic_th": topic["th"],
            "turns": conversations,
            "original_index": i,
            "success": True
        }
        
    except Exception as e:
        log_error_func("DeepSeekAPIError", i, f"Multi-turn API error: {e}. Prompt: {prompt[:200]}...")
        return {
            "context": sample_data["context"], 
            "topic_zh": topic["zh"],
            "topic_th": topic["th"],
            "turns": [],
            "original_index": i,
            "success": False,
            "error": str(e)
        }

def generate_multiturn_dialogues(n_samples: int = 50, seed: int = 42, max_workers: int = 3) -> pd.DataFrame:
    """Generate multi-turn dialogue dataset using DeepSeek AI with parallel processing"""
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in .env file")
    
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    
    random.seed(seed)
    print(f"[INFO] Preparing prompt combinations for {n_samples} multi-turn dialogue samples...")
    
    # Import templates
    try:
        from src.dataset_templates import MEDICAL_TOPICS, DIALOGUE_SAMPLES
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
        tasks_args.append((i, topic, sample, client, MULTI_TURN_PROMPT_TEMPLATE, logging.error))
    
    print(f"[INFO] Created {len(tasks_args)} tasks for multi-turn dialogue generation.")
    
    results = []
    successful_results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results_iterator = executor.map(_process_multiturn_sample, tasks_args)
        progress = ProgressBar(len(tasks_args), prefix='Generating Multi-turn Dialogues:', suffix='Complete')
        for result in results_iterator:
            results.append(result)
            if result["success"]:
                successful_results.append(result)
            progress.increment()

    # Process and flatten the results for DataFrame
    flattened_data = []
    
    for result in successful_results:
        for turn in result["turns"]:
            flattened_data.append({
                "context": result["context"],
                "topic_zh": result["topic_zh"],
                "topic_th": result["topic_th"],
                "turn_number": turn["turn"],
                "patient_zh": turn["patient_zh"],
                "patient_th": turn["patient_th"],
                "doctor_zh": turn["doctor_zh"],
                "doctor_th": turn["doctor_th"]
            })
    
    print(f"[INFO] Successfully generated {len(successful_results)} multi-turn dialogues with {len(flattened_data)} total turns.")
    
    df = pd.DataFrame(flattened_data)
    if not df.empty:
        df = add_metadata(df)
    return df
