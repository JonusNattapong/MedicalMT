"""
เตรียม dataset ใหม่ให้พร้อมและมีคุณภาพสำหรับ MedMT
- รวมข้อมูลจากไฟล์หลัก (dialogue, QA, reasoning, summary)
- ลบข้อมูลซ้ำและข้อมูลที่ขาดหาย
- สร้างไฟล์ clean_train.csv สำหรับเทรนโมเดล
"""
import os
import pandas as pd
def load_and_standardize(path, mapping, dropna_cols=None):
    df = pd.read_csv(path)
    df = df.rename(columns=mapping)
    if dropna_cols:
        df = df.dropna(subset=dropna_cols)
    # เลือกเฉพาะคอลัมน์ที่ต้องการ
    keep = [v for v in mapping.values() if v in df.columns]
    return df[keep]

def main():
    data_dir = "data"
    out_path = os.path.join(data_dir, "clean_train.csv")
    all_dfs = []
    # 1. Dialogue/translation
    for fname in ["train.csv", "deepseek_train.csv", "new_medical_dialogues_parallel_NDZ2T_20250511_225726.csv", "new_medical_dialogues_parallel_NHPVJ_20250511_224208.csv", "generated_train_dialogue_sample_GE7CKW_20250513231022.csv"]:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            df = load_and_standardize(fpath, {"context": "context", "source": "source", "target": "target"}, dropna_cols=["source", "target"])
            all_dfs.append(df)
    # 2. QA
    for fname in ["qa_train.csv", "qa_train_Q41PJ_20250513153116.csv", "qa_train_Q41PJ_20250513154720.csv", "qa_train_QIDOF_20250513_131903.csv"]:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            # ใช้ question_zh/answer_th เป็น source/target
            if "question_zh" in df.columns and "answer_th" in df.columns:
                qa_df = pd.DataFrame({
                    "context": df["context"] if "context" in df.columns else "",
                    "source": df["question_zh"],
                    "target": df["answer_th"]
                })
                qa_df = qa_df.dropna(subset=["source", "target"])
                all_dfs.append(qa_df)
    # 3. Reasoning
    fpath = os.path.join(data_dir, "reasoning.csv")
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        if "question_zh" in df.columns and "answer_th" in df.columns:
            reasoning_df = pd.DataFrame({
                "context": df["context"] if "context" in df.columns else "",
                "source": df["question_zh"],
                "target": df["answer_th"]
            })
            reasoning_df = reasoning_df.dropna(subset=["source", "target"])
            all_dfs.append(reasoning_df)
    # 4. Summary
    for fname in ["summary.csv", "summary_S88CD_20250511_143910.csv", "dataset_D88CD_20250511_144654.csv"]:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            if "summary_zh" in df.columns and "summary_th" in df.columns:
                summary_df = pd.DataFrame({
                    "context": df["context"] if "context" in df.columns else "",
                    "source": df["summary_zh"],
                    "target": df["summary_th"]
                })
                summary_df = summary_df.dropna(subset=["source", "target"])
                all_dfs.append(summary_df)
    # รวมและลบ duplicate
    if not all_dfs:
        print("ไม่พบไฟล์ข้อมูลที่ต้องการ!")
        return
    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=["context", "source", "target"])
    merged = merged.reset_index(drop=True)
    # ลบแถวที่ source/target เป็นค่าว่างหรือ null
    merged = merged.dropna(subset=["source", "target"])
    merged = merged[merged["source"].astype(str).str.strip() != ""]
    merged = merged[merged["target"].astype(str).str.strip() != ""]
    # บันทึกไฟล์ใหม่
    merged.to_csv(out_path, index=False)
    print(f"สร้างไฟล์ dataset ใหม่ที่ {out_path} จำนวน {len(merged)} ตัวอย่าง")

if __name__ == "__main__":
    main()
