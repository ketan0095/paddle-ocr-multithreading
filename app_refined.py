################################### Method 1: PDF to Image Conversion and OCR ###################################
################################### Original code with normal processing speed ##############################
import os
import json
import time
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from concurrent.futures import ProcessPoolExecutor
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








# Initialize PaddleOCR once in each process
def ocr_page(index, image_path):
    print(f'[Running Thread] : {index}')
    from paddleocr import PaddleOCR
    import json
    import numpy as np

    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        lang="en"
    )

    results = ocr.predict(image_path)

    # Convert results to JSON serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        # Add a fallback for unknown types:
        try:
            json.dumps(obj)
            return obj
        except TypeError:
            return str(obj)
        return obj

    serializable_results = convert_to_serializable(results)

    json_filename = image_path.replace(".png", "_result.json")
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)

    # Extract recognized texts
    page_text = []
    for res in results:
        if isinstance(res, list):
            # Each res is usually [bbox, (text, confidence)]
            page_text.append(res[1][0])

    return f"===== Page {index+1} =====\n" + "\n".join(page_text) + "\n"




if __name__ == "__main__":
    start_total = time.time()

    # Convert PDF to images
    print("[INFO] Converting PDF to images...")
    images = convert_from_path(PDF_PATH, dpi=300)

    image_paths = []
    for i, img in enumerate(images):
        img_path = os.path.join(OUTPUT_DIR, f"page_{i+1}.png")
        img.save(img_path, "PNG")
        image_paths.append((i, img_path))  # tuple of (index, path)

    # Parallel OCR - pass tuples directly
    print("os.cpu_count() :",os.cpu_count())
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        results = list(executor.map(ocr_page, *zip(*image_paths)))  # expands to two lists: indexes and paths

    all_text = "\n".join(results)
    with open(os.path.join(OUTPUT_DIR, "full_text.txt"), "w", encoding="utf-8") as f:
        f.write(all_text)

    print(f"[TIMER] Total time: {time.time() - start_total:.2f} sec")




