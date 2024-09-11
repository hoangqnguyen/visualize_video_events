import os
import pandas as pd
from utils import load_json, filter_events_by_score, non_max_suppression_events


# Function to process the JSON data into a CSV file format
def process_data(data, out_dir, scale_x=1.0, scale_y=1.0):
    max_len = 0
    for entry in data: # for each video
        event_dict = {}
        for event in entry["events"]:
            label = event["label"]
            event_dict[label] = event_dict.get(label, []) + [event["frame"]]
            event_dict[f"{label}_x"] = event_dict.get(f"{label}_x", []) + [int(event["xy"][0] * scale_x)]
            event_dict[f"{label}_y"] = event_dict.get(f"{label}_y", []) + [int(event["xy"][1] * scale_y)]
            
            max_len = max(max_len, len(event_dict[label]))
        
        # padding
        for label in event_dict:
            event_dict[label] += [""] * (max_len - len(event_dict[label]))
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, entry["video"] + ".csv").replace("2324", "").replace("_", "")
        pd.DataFrame(event_dict).to_csv(out_path, index=False)


if __name__ == "__main__":
    # Process the data

    preds = load_json("exp/pred-selected.eval_only.json")
    preds = filter_events_by_score(preds, 0.25)
    preds = non_max_suppression_events(preds, 5)

    process_data(preds, 'exp/whatever3', scale_x=224, scale_y=224)