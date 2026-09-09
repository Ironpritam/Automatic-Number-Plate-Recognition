import os
import time
import json
import pandas as pd
from typing import Dict, Any, Optional

class ANPRLogger:
    """
    Structured logger for recorded ANPR detections.
    Supports CSV logging (`data.csv`) and JSON audit trailing.
    """
    def __init__(self, csv_path: str = "data.csv", json_log_dir: str = "logs"):
        self.csv_path = csv_path
        self.json_log_dir = json_log_dir
        os.makedirs(self.json_log_dir, exist_ok=True)

    def log_detection(self, plate_text: str, confidence: Optional[float] = None, language: str = "eng") -> Dict[str, Any]:
        """Logs a detected number plate to CSV database and JSON log record."""
        current_time = time.asctime(time.localtime(time.time()))
        timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        
        record = {
            "timestamp": current_time,
            "iso_timestamp": timestamp_iso,
            "plate_number": plate_text,
            "language": language,
            "confidence": confidence if confidence is not None else 0.0
        }

        # CSV Logging
        if len(plate_text) >= 4 and plate_text != "Not Detected":
            self._update_csv(current_time, plate_text)
            self._write_json(record)

        return record

    def _update_csv(self, timestamp: str, plate_text: str):
        """Updates or appends to data.csv file."""
        if not os.path.exists(self.csv_path):
            df = pd.DataFrame([{"date": timestamp, "Number Plate": plate_text}])
            df.to_csv(self.csv_path, index=False)
        else:
            df = pd.read_csv(self.csv_path)
            # Match existing plate entry
            match = df[df['Number Plate'] == plate_text]
            if not match.empty:
                idx = match.index[0]
                df.loc[idx, 'date'] = timestamp
            else:
                new_row = pd.DataFrame([{"date": timestamp, "Number Plate": plate_text}])
                df = pd.concat([df, new_row], ignore_index=True)
            
            df.to_csv(self.csv_path, index=False)

    def _write_json(self, record: Dict[str, Any]):
        """Writes JSON audit trail entry."""
        filename = f"detection_{int(time.time())}.json"
        filepath = os.path.join(self.json_log_dir, filename)
        with open(filepath, "w") as f:
            json.dump(record, f, indent=2)
