################################### Method 1: PDF to Image Conversion and OCR ###################################
################################### Original code with normal processing speed ##############################
import os
import json
import time
from paddleocr import PaddleOCR
from pdf2image import convert_from_path

# --- CONFIG ---
DOC_FOLDER='./docs'
PDF_PATH = os.path.join(DOC_FOLDER,"input_agreement.pdf")  # Path to your PDF
OUTPUT_DIR = "ocr_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Initialize PaddleOCR ---
ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en"
)

start_total = time.time()

# --- Convert PDF to images ---
t1 = time.time()
print("[INFO] Converting PDF to images...")
images = convert_from_path(PDF_PATH, dpi=300)
t2 = time.time()
print(f"[TIMER] PDF to images: {t2 - t1:.2f} sec")

all_text = []

# --- OCR each page ---
t3 = time.time()
for i, img in enumerate(images):
    img_path = os.path.join(OUTPUT_DIR, f"page_{i+1}.png")
    img.save(img_path, "PNG")
    print(f"[INFO] Saved {img_path}")

    print(f"[INFO] Running OCR on page {i+1}...")
    results = ocr.predict(img_path)

    page_text = []
    for res in results:
        json_filename = img_path.replace(".png", "_result.json")
        res.save_to_json(json_filename)

        with open(json_filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
            if 'res' in json_data and 'rec_texts' in json_data['res']:
                page_text = json_data['res']['rec_texts']
            elif 'rec_texts' in json_data:
                page_text = json_data['rec_texts']

    all_text.append(f"===================================> Page {i+1} <=====================================\n" + "\n".join(page_text) + "\n")

t4 = time.time()
print(f"[TIMER] OCR processing: {t4 - t3:.2f} sec")

# --- Output ---
print("\n================================> OCR RESULT <=======================================\n")
# print("".join(all_text))

# Save to file
output_text_file = os.path.join(OUTPUT_DIR, "extracted_text.txt")
with open(output_text_file, "w", encoding="utf-8") as f:
    f.writelines(all_text)

end_total = time.time()
print(f"[TIMER] TOTAL TIME: {end_total - start_total:.2f} sec")
print(f"[INFO] OCR result saved to {output_text_file}")

