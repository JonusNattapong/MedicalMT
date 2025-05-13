# -*- coding: utf-8 -*-
"""
Text processing utilities for MedMT dataset generation
"""
import re
import random

def extract_thai_translation(response_text):
    """Extract only the Thai translation from DeepSeek response"""
    # Try to find text between quotes after "Translation:"
    match = re.search(r'Translation:?\s*[\""]([^\""]+)[\""]', response_text)
    if match:
        return match.group(1).strip()
    
    # If no quotes found, try to find Thai text (characters in Thai unicode range)
    thai_text_segments = re.findall(r'[\u0E00-\u0E7F]+(?:[\s\u0E00-\u0E7F]*[\u0E00-\u0E7F]+)*[\?]?', response_text)
    
    if thai_text_segments:
        # Join all found Thai segments
        combined = " ".join(segment.strip() for segment in thai_text_segments if segment.strip()).strip()
        
        # Ensure question mark if source ends with 吗
        if "吗" in response_text and not combined.endswith("?"):
            combined += "?"
        return combined
    
    return ""

def _create_diverse_prompt(topic, sample, index, prompt_template):
    """Creates a more diverse prompt by adding small variations"""
    context_enhancers = [
        "请注意以下对话内容：", 
        "请查看以下医疗对话：", 
        "以下是医生和病人的对话：",
        "分析以下病例：",
        "请阅读以下医疗咨询：",
        "医疗记录如下：",
        "患者咨询内容：",
        "就诊记录：",
        "医患沟通记录：",
        "门诊记录摘要：",
        "临床诊断对话：",
        ""  # Include option for no enhancement
    ]
    
    # Add variation based on index
    enhancer_index = hash(str(index) + "enhancer_salt") % len(context_enhancers)
    enhancer = context_enhancers[enhancer_index]
    
    # Add uniqueness with case number
    case_id = f"(案例 #{index + 1})"
    
    description_enhancers = [
        f"{topic['desc']} {case_id}",
        f"{topic['desc']} - 常见于{case_id}",
        f"{topic['desc']} (重要医疗情况) {case_id}",
        f"{topic['desc']} - 需要专业翻译 {case_id}",
        f"{topic['desc']} - 医学专业术语 {case_id}"
    ]
    
    desc_index = hash(str(index) + "desc_salt") % len(description_enhancers)
    enhanced_desc = description_enhancers[desc_index]
    
    # Format the prompt with variations
    prompt = prompt_template.format(
        context=(f"{enhancer}\n{sample['context']}" if enhancer else sample["context"]),
        topic_zh=topic["zh"],
        topic_th=topic["th"],
        topic_desc=enhanced_desc,
        symptom=sample["symptom"],
        source=sample["source"]
    )
    
    return prompt

def _get_dynamic_parameters(index, base_temp=0.7):
    """Generate dynamic parameters for the API call to increase diversity"""
    index_hash = hash(str(index) + "params_salt")
    
    # Temperature variation
    temp_variation = (index_hash % 5) * 0.05
    temperature = base_temp + temp_variation
    temperature = min(0.95, max(0.5, temperature))
    
    # Penalty variations
    presence_penalty = 0.0 + (index_hash % 5) * 0.05
    frequency_penalty = 0.0 + (index_hash % 4) * 0.05
    
    # Top_p variation
    top_p = 0.90 + (index_hash % 5) * 0.02
    top_p = min(0.98, max(0.85, top_p))
    
    return {
        "temperature": temperature,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "top_p": top_p
    }