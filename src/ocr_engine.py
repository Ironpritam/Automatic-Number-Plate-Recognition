import os
import re
import shutil
import cv2
import numpy as np
import pytesseract
from .preprocessing import Preprocessing

# Pure Python Devanagari to Latin Digit & Character Mapper (100% Offline)
DEVANAGARI_DIGIT_MAP = {
    '०': '0', '१': '1', '२': '2', '३': '3', '४': '4',
    '५': '5', '६': '6', '७': '7', '८': '8', '९': '9'
}

DEVANAGARI_STATE_MAP = {
    'महाराष्ट्र': 'MH',
    'एमएच': 'MH',
    'एम एच': 'MH',
    'म.रा.': 'MH',
    'मरा': 'MH'
}

class ANPROCREngine:
    """
    Multi-lingual OCR Engine for Indian Number Plate Recognition.
    Supports English & Marathi Devanagari scripts with dynamic Tesseract path resolution 
    and offline fallback parsing.
    """
    def __init__(self, tesseract_cmd: str = None):
        self.setup_tesseract(tesseract_cmd)

    def setup_tesseract(self, custom_path: str = None):
        """Resolves Tesseract binary path across OS platforms dynamically."""
        if custom_path and os.path.exists(custom_path):
            pytesseract.pytesseract.tesseract_cmd = custom_path
            return

        env_path = os.environ.get("TESSERACT_CMD")
        if env_path and os.path.exists(env_path):
            pytesseract.pytesseract.tesseract_cmd = env_path
            return

        system_tesseract = shutil.which("tesseract")
        if system_tesseract:
            pytesseract.pytesseract.tesseract_cmd = system_tesseract
            return

        # Common Windows Installation Paths Fallback
        default_windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe")
        ]
        for path in default_windows_paths:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return

    def parse_devanagari_offline(self, text: str) -> str:
        """Transliterates Marathi Devanagari digits and state prefixes to standard Latin plate text."""
        result = text
        for dev_str, lat_str in DEVANAGARI_STATE_MAP.items():
            result = result.replace(dev_str, lat_str)

        for dev_digit, lat_digit in DEVANAGARI_DIGIT_MAP.items():
            result = result.replace(dev_digit, lat_digit)

        # Retain alphanumeric characters only
        clean_text = re.sub(r'[^a-zA-Z0-9]', '', result)

        # Post-process state prefix for Maharashtra plates if numbers detected
        if len(clean_text) >= 4 and clean_text.isdigit():
            clean_text = "MH" + clean_text
        elif not clean_text.startswith("MH") and any(c.isdigit() for c in clean_text):
            # Check if leading letters exist, else prefix MH
            first_digit_idx = next((i for i, c in enumerate(clean_text) if c.isdigit()), None)
            if first_digit_idx is not None and first_digit_idx < 2:
                clean_text = "MH" + clean_text[first_digit_idx:]

        return clean_text

    def recognize_english(self, roi_img: np.ndarray) -> str:
        """Extracts text from English number plate ROI."""
        preprocessed = Preprocessing(roi_img)
        config = '--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        raw_text = pytesseract.image_to_string(preprocessed, config=config, lang="eng")
        clean_text = re.sub(r'[^a-zA-Z0-9]', '', raw_text)
        return clean_text

    def recognize_marathi(self, roi_img: np.ndarray) -> str:
        """Extracts text from Marathi number plate ROI with offline Devanagari mapping & Google Translate fallback."""
        if roi_img is None or roi_img.size == 0:
            return "Not Detected"

        blur = cv2.GaussianBlur(roi_img, (3, 3), 0)
        filtered = cv2.bilateralFilter(blur, 11, 17, 17)
        gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY) if len(filtered.shape) == 3 else filtered

        options = "-l mar --psm 6"
        try:
            raw_text = pytesseract.image_to_string(gray, config=options)
            
            # First try offline parsing
            offline_result = self.parse_devanagari_offline(raw_text)
            if len(offline_result) >= 4:
                return offline_result

            # Online googletrans fallback if available
            try:
                from googletrans import Translator
                translator = Translator()
                translated = translator.translate(raw_text, src="mr")
                trans_text = re.sub(r'[^a-zA-Z0-9]', '', translated.text)
                if len(trans_text) >= 4:
                    return trans_text
            except Exception:
                pass

            return offline_result if offline_result else "Not Detected"
        except Exception as e:
            return "Not Detected"

    def process_roi(self, roi_img: np.ndarray, lang: str = "eng") -> str:
        """Main interface for ROI character recognition."""
        if lang in ["mr", "marathi"]:
            return self.recognize_marathi(roi_img)
        return self.recognize_english(roi_img)
