# Scripts Directory

โครงสร้างของโฟลเดอร์สคริปต์:

- `data/` - สคริปต์สำหรับการจัดการข้อมูล
  - generate_full_dataset.py - สคริปต์สร้างชุดข้อมูลฉบับสมบูรณ์

- `metrics/` - สคริปต์สำหรับการวัดประสิทธิภาพ
  - compare_metrics.py - สคริปต์เปรียบเทียบค่าประสิทธิภาพต่างๆ

- `utils/` - สคริปต์อรรถประโยชน์
  - simple_xmodel_test.py - สคริปต์ทดสอบโมเดลอย่างง่าย
  - run_medmt_workflow.sh - สคริปต์รันขั้นตอนการทำงานทั้งหมด
  - run_xmodel_test.sh - สคริปต์รันการทดสอบโมเดล

การใช้งาน:
1. รันการทดสอบ: `./utils/run_xmodel_test.sh`
2. รันขั้นตอนทั้งหมด: `./utils/run_medmt_workflow.sh`