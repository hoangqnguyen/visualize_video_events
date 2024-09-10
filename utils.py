import json
import gzip


def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)


def load_gz_json(fpath):
    with gzip.open(fpath, "rt", encoding="ascii") as fp:
        return json.load(fp)


def store_json(fpath, obj, pretty=False):
    kwargs = {}
    if pretty:
        kwargs["indent"] = 2
        kwargs["sort_keys"] = True
    with open(fpath, "w") as fp:
        json.dump(obj, fp, **kwargs)


def store_gz_json(fpath, obj):
    with gzip.open(fpath, "wt", encoding="ascii") as fp:
        json.dump(obj, fp)


def filter_events_by_score(data, fg_threshold):
    filtered_data = []
    for video in data:
        filtered_events = [
            event for event in video["events"] if event["score"] >= fg_threshold
        ]
        filtered_video = {
            "video": video["video"],
            "events": filtered_events,
            "fps": video["fps"],
        }
        filtered_data.append(filtered_video)
    return filtered_data


def non_max_suppression_events(data, tol_t):
    def suppress_events(events, tol_t):
        events.sort(key=lambda x: x["frame"])
        suppressed_events = []
        i = 0
        while i < len(events):
            current_event = events[i]
            j = i + 1
            while (
                j < len(events) and events[j]["frame"] - current_event["frame"] <= tol_t
            ):
                if events[j]["score"] > current_event["score"]:
                    current_event = events[j]
                j += 1
            suppressed_events.append(current_event)
            i = j
        return suppressed_events

    for video in data:
        events_by_label = {}
        for event in video["events"]:
            label = event["label"]
            if label not in events_by_label:
                events_by_label[label] = []
            events_by_label[label].append(event)

        suppressed_events = []
        for label, events in events_by_label.items():
            suppressed_events.extend(suppress_events(events, tol_t))

        video["events"] = suppressed_events

    return data
