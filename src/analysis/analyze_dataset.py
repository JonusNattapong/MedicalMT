import pandas as pd
import os
import sys

# Default ตำแหน่งไฟล์
default_file_path = r'C:\Users\Admin\OneDrive\เอกสาร\Github\MedMT\data\new_medical_dialogues_parallel_NDZ2T_20250511_225726.csv'
output_file = r'C:\Users\Admin\OneDrive\เอกสาร\Github\MedMT\data_analysis_result.txt'

# ตรวจสอบว่ามีการระบุไฟล์ผ่าน command line หรือไม่
if len(sys.argv) > 1:
    file_path = sys.argv[1]
else:
    file_path = default_file_path

# อ่านไฟล์ CSV
df = pd.read_csv(file_path)

# เตรียมข้อความสำหรับเขียนลงไฟล์
results = []
results.append(f"=== การวิเคราะห์ข้อมูล: {os.path.basename(file_path)} ===")
results.append(f"จำนวนรายการทั้งหมด: {len(df)}")
results.append(f"คอลัมน์ในข้อมูล: {', '.join(df.columns.tolist())}")

# นับจำนวนข้อมูลที่ไม่ซ้ำกัน
results.append(f"\n=== ความหลากหลายของข้อมูล ===")
results.append(f"จำนวนประโยคต้นฉบับ (source) ที่ไม่ซ้ำกัน: {df['source'].nunique()} จาก {len(df)} ({df['source'].nunique()/len(df)*100:.2f}%)")
results.append(f"จำนวนคำแปล (target) ที่ไม่ซ้ำกัน: {df['target'].nunique()} จาก {len(df)} ({df['target'].nunique()/len(df)*100:.2f}%)")
results.append(f"จำนวนบริบท (context) ที่ไม่ซ้ำกัน: {df['context'].nunique()} จาก {len(df)} ({df['context'].nunique()/len(df)*100:.2f}%)")

# ดูจำนวนการใช้ซ้ำของประโยคต้นฉบับ
source_counts = df['source'].value_counts()
results.append(f"\n=== การใช้ซ้ำของประโยคต้นฉบับ (source) ===")
results.append(f"จำนวนประโยคที่ปรากฏเพียงครั้งเดียว: {(source_counts == 1).sum()}")
results.append(f"จำนวนประโยคที่ถูกใช้ซ้ำมากกว่า 1 ครั้ง: {(source_counts > 1).sum()}")
results.append(f"จำนวนการใช้ซ้ำสูงสุด: {source_counts.max()} ครั้ง")
results.append(f"ประโยคที่ถูกใช้ซ้ำมากที่สุด 5 อันดับแรก:")
for source, count in source_counts.nlargest(5).items():
    results.append(f"  - '{source}': {count} ครั้ง")

# ดูตัวอย่างคำแปลที่แตกต่างกันสำหรับประโยคเดียวกัน
results.append(f"\n=== ตัวอย่างความหลากหลายของคำแปล ===")
most_common_source = source_counts.idxmax()
different_translations = df[df['source'] == most_common_source][['target']].drop_duplicates()
results.append(f"ตัวอย่างคำแปลที่แตกต่างกันสำหรับประโยค '{most_common_source}':")
for i, translation in enumerate(different_translations['target'].head(5).tolist(), 1):
    results.append(f"  {i}. {translation}")

results.append("\n=== สรุป ===")
diversity_score = (df['source'].nunique() + df['target'].nunique() + df['context'].nunique()) / (3 * len(df)) * 100
results.append(f"คะแนนความหลากหลายของข้อมูล: {diversity_score:.2f}%")
if diversity_score > 80:
    status = "ดีมาก"
elif diversity_score > 60:
    status = "ดี"
elif diversity_score > 40:
    status = "พอใช้"
else:
    status = "ควรปรับปรุง"
results.append(f"สถานะความหลากหลาย: {status}")

# เขียนผลลัพธ์ลงไฟล์
with open(output_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(results))

# แสดงผลลัพธ์บนหน้าจอด้วย
for line in results:
    print(line)

print(f"\nผลการวิเคราะห์ถูกบันทึกลงในไฟล์: {output_file}")
