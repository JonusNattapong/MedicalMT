# Documentation: MedMT Medical Dialogue Machine Translation

## แนวคิดและหลักการ
- ใช้โมเดล Neural Machine Translation (NMT) ที่รองรับ context-aware translation
- เน้นความถูกต้องทางการแพทย์และความลื่นไหลของบทสนทนา
- สนับสนุนการขยาย dataset ด้วย synthetic data
- ใช้เทคนิค LoRA/PEFT, DeepSpeed, Accelerate สำหรับการ train ขนาดใหญ่

## วิธีการทำงาน
1. เตรียมข้อมูล (train/test) ในรูปแบบ context, source, target
2. Preprocess ข้อมูลให้เหมาะกับโมเดล (context-aware)
3. Train ด้วย Huggingface Transformers (รองรับ context)
4. Save checkpoint ทั้งแบบ PyTorch และ .safetensors
5. Evaluate ด้วย BLEU และ human review
6. Inference และสร้างไฟล์ submission
7. อัปโหลดโมเดล/ชุดข้อมูลขึ้น Hugging Face Hub

## การใช้งานฟีเจอร์ขั้นสูง
- ใช้ LoRA/PEFT เพื่อลด resource ในการ fine-tune
- DeepSpeed/Accelerate สำหรับ multi-GPU/TPU
- safetensors สำหรับ checkpoint ที่ปลอดภัยและเร็ว
- Hugging Face Hub สำหรับแชร์โมเดล/ข้อมูล

## License
- โค้ด โมเดล และข้อมูล อยู่ภายใต้ CC BY-SA-NC 4.0 (ใช้เพื่อการวิจัย/ไม่เชิงพาณิชย์เท่านั้น)

## อื่นๆ
- สามารถปรับแต่ง config.yaml เพื่อเปลี่ยน hyperparameters/model
- ตัวอย่างการ push model/dataset ดูใน README.md

---

> MedMT © 2025 | CC BY-SA-NC 4.0
