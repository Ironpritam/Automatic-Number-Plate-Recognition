import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Any, Tuple

# Path resolution for YOLOv5 modules
FILE = Path(__file__).resolve()
SRC_DIR = FILE.parent
ROOT_DIR = SRC_DIR.parent
YOLOV5_DIR = ROOT_DIR / "yolov5"

if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
if str(YOLOV5_DIR) not in sys.path:
    sys.path.append(str(YOLOV5_DIR))

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords, save_one_box
from utils.torch_utils import select_device

from .ocr_engine import ANPROCREngine
from .logger import ANPRLogger

class ANPRDetector:
    """
    High-level Object Detection & Recognition Interface for Indian License Plates.
    Combines fine-tuned YOLOv5 model bounding box inference with ANPROCREngine.
    """
    def __init__(self, weights_path: str = None, device_str: str = ''):
        self.root_dir = ROOT_DIR
        self.weights_path = weights_path if weights_path else str(self.root_dir / 'weights' / 'best.pt')
        self.device = select_device(device_str)
        self.model = attempt_load(self.weights_path, map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.img_size = check_img_size(640, s=self.stride)
        self.model.eval()

        self.ocr_engine = ANPROCREngine()
        self.logger = ANPRLogger(csv_path=str(self.root_dir / "data" / "data.csv"))

    def detect_and_recognize(
        self, 
        image_input: Any, 
        conf_thres: float = 0.25, 
        iou_thres: float = 0.45, 
        lang: str = 'eng'
    ) -> Dict[str, Any]:
        """
        Runs object detection on input image (file path or numpy array), crops plate ROI,
        and applies OCR recognition.
        """
        if isinstance(image_input, (str, Path)):
            img0 = cv2.imread(str(image_input))
        elif isinstance(image_input, np.ndarray):
            img0 = image_input.copy()
        else:
            raise ValueError("Unsupported image input type. Provide file path or numpy ndarray.")

        if img0 is None:
            return {"status": "error", "message": "Failed to decode image input."}

        # Letterbox resizing for YOLOv5 model
        img = self._preprocess_for_yolo(img0)
        img = torch.from_numpy(img).to(self.device).float() / 255.0
        if len(img.shape) == 3:
            img = img[None]

        # Model Inference
        with torch.no_grad():
            pred = self.model(img)[0]

        # Apply Non-Maximum Suppression (NMS)
        pred = non_max_suppression(pred, conf_thres, iou_thres)

        detected_plate_text = "Not Detected"
        max_conf = 0.0
        cropped_roi = None

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    confidence_score = float(conf)
                    if confidence_score >= conf_thres:
                        crop = save_one_box(
                            xyxy, img0.copy(), 
                            file=self.root_dir / 'uploads/cropped.jpg', 
                            BGR=True, save=False
                        )
                        if crop is not None and crop.size > 0:
                            cropped_roi = crop
                            recognized_text = self.ocr_engine.process_roi(crop, lang=lang)
                            if len(recognized_text) >= 3 and confidence_score > max_conf:
                                max_conf = confidence_score
                                detected_plate_text = recognized_text

        # Log Detection Result
        logged_record = self.logger.log_detection(
            plate_text=detected_plate_text,
            confidence=max_conf,
            language=lang
        )

        return {
            "status": "success",
            "detected_plate": detected_plate_text,
            "confidence": max_conf,
            "language": lang,
            "timestamp": logged_record["timestamp"]
        }

    def _preprocess_for_yolo(self, img0: np.ndarray) -> np.ndarray:
        """Resizes and pads image to fit YOLOv5 stride requirement (640x640)."""
        h0, w0 = img0.shape[:2]
        r = self.img_size / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
            img = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)
        else:
            img = img0

        h, w = img.shape[:2]
        dh, dw = self.img_size - h, self.img_size - w
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)

        padded_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        padded_img = padded_img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        padded_img = np.ascontiguousarray(padded_img)
        return padded_img
