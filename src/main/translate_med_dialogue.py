import argparse
import os
import pandas as pd
from datasets import load_metric, Dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,
                          Seq2SeqTrainingArguments, Seq2SeqTrainer)
import torch

MODEL_NAME = "XiaoduoAILab/XmodelLM1.5" 


def load_data(train_path, test_path, context_col="Context", source_col="Source", target_col="Target"):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def preprocess_function(examples, tokenizer, context_col, source_col, target_col=None, max_length=256):
    src_texts = [
        (c + "\n" if c and not pd.isna(c) else "") + s
        for c, s in zip(examples[context_col], examples[source_col])
    ]
    model_inputs = tokenizer(src_texts, max_length=max_length, truncation=True)
    if target_col and target_col in examples:
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(examples[target_col], max_length=max_length, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def train_model(train_file, output_dir, context_col="Context", source_col="Source", target_col="Target"):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
    train_df = pd.read_csv(train_file)
    train_dataset = Dataset.from_pandas(train_df)
    tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer, context_col, source_col, target_col), batched=True)
    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="no",
        per_device_train_batch_size=8,
        num_train_epochs=3,
        save_total_limit=1,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        logging_steps=50,
        save_steps=500,
        learning_rate=5e-5,
        report_to=[],
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)


def evaluate_model(model_dir, test_file, context_col="Context", source_col="Source", target_col="Target"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=True)
    test_df = pd.read_csv(test_file)
    test_dataset = Dataset.from_pandas(test_df)
    metric = load_metric("sacrebleu")
    preds = []
    refs = []
    for i, row in test_df.iterrows():
        src = (str(row[context_col]) + "\n" if context_col in row and pd.notna(row[context_col]) else "") + str(row[source_col])
        input_ids = tokenizer(src, return_tensors="pt", truncation=True, max_length=256).input_ids
        with torch.no_grad():
            output = model.generate(input_ids, max_length=256, num_beams=4)
        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        preds.append(pred)
        refs.append([str(row[target_col])])
    bleu = metric.compute(predictions=preds, references=refs)["score"]
    print(f"BLEU score: {bleu:.2f}")
    return bleu


def translate_file(model_dir, input_file, output_file, context_col="Context", source_col="Source"):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, trust_remote_code=True)
    df = pd.read_csv(input_file)
    translations = []
    for i, row in df.iterrows():
        src = (str(row[context_col]) + "\n" if context_col in row and pd.notna(row[context_col]) else "") + str(row[source_col])
        input_ids = tokenizer(src, return_tensors="pt", truncation=True, max_length=256).input_ids
        with torch.no_grad():
            output = model.generate(input_ids, max_length=256, num_beams=4)
        pred = tokenizer.decode(output[0], skip_special_tokens=True)
        translations.append(pred)
    df["Prediction"] = translations
    df.to_csv(output_file, index=False)
    print(f"Saved translations to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Medical Dialogue Machine Translation (Chinese→Thai)")
    parser.add_argument("--train", type=str, help="Path to training CSV file")
    parser.add_argument("--test", type=str, help="Path to test CSV file")
    parser.add_argument("--output_dir", type=str, default="./mt_model", help="Directory to save model")
    parser.add_argument("--do_train", action="store_true", help="Train the model")
    parser.add_argument("--do_eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--do_translate", action="store_true", help="Translate test file")
    parser.add_argument("--output_file", type=str, default="predictions.csv", help="Output file for translations")
    args = parser.parse_args()

    if args.do_train:
        train_model(args.train, args.output_dir)
    if args.do_eval:
        evaluate_model(args.output_dir, args.test)
    if args.do_translate:
        translate_file(args.output_dir, args.test, args.output_file)

if __name__ == "__main__":
    main()

