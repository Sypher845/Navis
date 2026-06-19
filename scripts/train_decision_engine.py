"""
train_decision_engine.py — Train a decision tree that learns optimal
traffic management actions from historical incident outcomes.

v2: Adds class weighting, junction/police station statistics,
    and corridor co-occurrence matrix for smart diversions.

Target classes (derived from data):
  0 = monitor        — Low priority, no closure
  1 = deploy_light   — High priority, no closure
  2 = deploy_heavy   — High priority + road closure
  3 = full_closure   — Event-type cause + road closure

Usage: python scripts/train_decision_engine.py
Output: public/decision_tree.json
"""

import csv
import json
import os
import math
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict

INPUT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
)
OUTPUT_JSON = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'public', 'decision_tree.json'
)

EVENT_CAUSES = {'public_event', 'procession', 'vip_movement', 'protest'}

ACTION_LABELS = ['monitor', 'deploy_light', 'deploy_heavy', 'full_closure']

ACTION_DESCRIPTIONS = {
    'monitor': 'Monitor situation -- likely self-resolving',
    'deploy_light': 'Light deployment -- traffic police + cones',
    'deploy_heavy': 'Heavy deployment -- barricades + diversions + tow trucks',
    'full_closure': 'Full closure protocol -- pre-emptive road block + diversion plan',
}


# --- Data loading and labeling ---

def parse_ist(dt_str):
    if not dt_str or dt_str == 'NULL':
        return None
    try:
        clean = dt_str.split('+')[0].split('.')[0]
        dt = datetime.strptime(clean, '%Y-%m-%d %H:%M:%S')
        return dt + timedelta(hours=5, minutes=30)
    except Exception:
        return None


def compute_resolution_min(row):
    start = parse_ist(row['start_datetime'])
    end = parse_ist(row.get('closed_datetime', '') or row.get('resolved_datetime', ''))
    if not start or not end:
        return None
    mins = (end - start).total_seconds() / 60
    return mins if 0 < mins < 1440 else None


def derive_action_label(row, resolution_min):
    """Derive what action was effectively taken, from the outcome data."""
    is_high = row['priority'] == 'High'
    has_closure = row['requires_road_closure'] == 'TRUE'
    is_event = row['event_cause'] in EVENT_CAUSES

    if is_event and has_closure:
        return 3  # full_closure
    elif is_high and has_closure:
        return 2  # deploy_heavy
    elif is_high:
        return 1  # deploy_light
    else:
        return 0  # monitor


def safe_str(val):
    if not val or val == 'NULL' or val.strip() == '':
        return None
    return val.strip()


def load_dataset():
    """Load CSV and build feature matrix + labels."""
    records = []
    with open(INPUT_CSV, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            dt = parse_ist(row['start_datetime'])
            if not dt:
                continue

            lat = float(row['latitude']) if row['latitude'] and row['latitude'] != 'NULL' else None
            lng = float(row['longitude']) if row['longitude'] and row['longitude'] != 'NULL' else None
            if not lat or not lng:
                continue

            res_min = compute_resolution_min(row)
            label = derive_action_label(row, res_min)

            corridor = row['corridor'] if row['corridor'] != 'NULL' else 'Non-corridor'
            zone = row['zone'] if row['zone'] != 'NULL' else 'Unknown'
            junction = safe_str(row.get('junction', ''))
            police_station = safe_str(row.get('police_station', ''))
            veh_type = safe_str(row.get('veh_type', ''))

            records.append({
                'cause': row['event_cause'],
                'corridor': corridor,
                'zone': zone,
                'hour': dt.hour,
                'day_of_week': dt.weekday(),
                'event_type': row['event_type'],
                'lat': lat,
                'lng': lng,
                'resolution_min': res_min,
                'label': label,
                'junction': junction,
                'police_station': police_station,
                'veh_type': veh_type,
                'road_closure': row['requires_road_closure'] == 'TRUE',
                'priority': row['priority'],
                'date_key': dt.strftime('%Y-%m-%d'),
            })

    return records


# --- Decision Tree (with class weighting) ---

class DecisionNode:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.categories = None
        self.left = None
        self.right = None
        self.prediction = None
        self.distribution = None
        self.samples = 0
        self.reason = ''
        self.outcome_stats = None


def weighted_gini(labels, class_weights):
    """Compute weighted Gini impurity."""
    n_weighted = sum(class_weights.get(l, 1.0) for l in labels)
    if n_weighted == 0:
        return 0
    counts = Counter(labels)
    impurity = 1.0
    for cls, count in counts.items():
        w = class_weights.get(cls, 1.0)
        p = (count * w) / n_weighted
        impurity -= p * p
    return impurity


def weighted_info_gain(parent_labels, left_labels, right_labels, class_weights):
    """Weighted Gini decrease."""
    n_parent = sum(class_weights.get(l, 1.0) for l in parent_labels)
    if n_parent == 0:
        return 0
    n_left = sum(class_weights.get(l, 1.0) for l in left_labels)
    n_right = sum(class_weights.get(l, 1.0) for l in right_labels)
    parent_gini = weighted_gini(parent_labels, class_weights)
    left_gini = weighted_gini(left_labels, class_weights)
    right_gini = weighted_gini(right_labels, class_weights)
    weighted = (n_left / n_parent) * left_gini + (n_right / n_parent) * right_gini
    return parent_gini - weighted


def best_split_categorical(records, feature, labels, class_weights):
    values = set(r[feature] for r in records)
    if len(values) <= 1:
        return None, None, 0

    best_gain = 0
    best_cats = None

    for val in values:
        left_idx = [i for i, r in enumerate(records) if r[feature] == val]
        right_idx = [i for i, r in enumerate(records) if r[feature] != val]
        if len(left_idx) == 0 or len(right_idx) == 0:
            continue
        gain = weighted_info_gain(
            labels,
            [labels[i] for i in left_idx],
            [labels[i] for i in right_idx],
            class_weights,
        )
        if gain > best_gain:
            best_gain = gain
            best_cats = {val}

    if feature == 'cause':
        left_idx = [i for i, r in enumerate(records) if r[feature] in EVENT_CAUSES]
        right_idx = [i for i, r in enumerate(records) if r[feature] not in EVENT_CAUSES]
        if len(left_idx) > 0 and len(right_idx) > 0:
            gain = weighted_info_gain(
                labels,
                [labels[i] for i in left_idx],
                [labels[i] for i in right_idx],
                class_weights,
            )
            if gain > best_gain:
                best_gain = gain
                best_cats = EVENT_CAUSES

    return best_cats, None, best_gain


def best_split_numeric(records, feature, labels, class_weights):
    values = sorted(set(r[feature] for r in records))
    if len(values) <= 1:
        return None, None, 0

    best_gain = 0
    best_threshold = None

    candidates = values
    if len(candidates) > 20:
        step = max(1, len(candidates) // 20)
        candidates = candidates[::step]

    for i in range(len(candidates) - 1):
        threshold = (candidates[i] + candidates[i + 1]) / 2
        left_idx = [j for j, r in enumerate(records) if r[feature] <= threshold]
        right_idx = [j for j, r in enumerate(records) if r[feature] > threshold]
        if len(left_idx) == 0 or len(right_idx) == 0:
            continue
        gain = weighted_info_gain(
            labels,
            [labels[j] for j in left_idx],
            [labels[j] for j in right_idx],
            class_weights,
        )
        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return None, best_threshold, best_gain


CATEGORICAL_FEATURES = ['cause', 'corridor', 'zone', 'event_type']
NUMERIC_FEATURES = ['hour', 'day_of_week', 'lat', 'lng']
DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def build_tree(records, labels, class_weights, depth=0, max_depth=7, min_samples=10):
    node = DecisionNode()
    node.samples = len(records)
    node.distribution = dict(Counter(labels))

    res_times = [r['resolution_min'] for r in records if r['resolution_min'] is not None]
    node.outcome_stats = {
        'median_resolution_min': round(float(np.median(res_times)), 1) if res_times else None,
        'count_with_resolution': len(res_times),
    }

    if depth >= max_depth or len(records) < min_samples or len(set(labels)) <= 1:
        # Weighted majority vote
        weighted_counts = {}
        for l in set(labels):
            weighted_counts[l] = labels.count(l) * class_weights.get(l, 1.0)
        node.prediction = max(weighted_counts, key=weighted_counts.get)
        return node

    best_feature = None
    best_cats = None
    best_threshold = None
    best_gain = 0

    for feat in CATEGORICAL_FEATURES:
        cats, _, gain = best_split_categorical(records, feat, labels, class_weights)
        if gain > best_gain:
            best_gain = gain
            best_feature = feat
            best_cats = cats
            best_threshold = None

    for feat in NUMERIC_FEATURES:
        _, threshold, gain = best_split_numeric(records, feat, labels, class_weights)
        if gain > best_gain:
            best_gain = gain
            best_feature = feat
            best_cats = None
            best_threshold = threshold

    if best_gain < 0.001 or best_feature is None:
        weighted_counts = {}
        for l in set(labels):
            weighted_counts[l] = labels.count(l) * class_weights.get(l, 1.0)
        node.prediction = max(weighted_counts, key=weighted_counts.get)
        return node

    node.feature = best_feature

    if best_cats is not None:
        node.categories = list(best_cats)
        left_idx = [i for i, r in enumerate(records) if r[best_feature] in best_cats]
        right_idx = [i for i, r in enumerate(records) if r[best_feature] not in best_cats]
        cat_label = ', '.join(sorted(best_cats))
        node.reason = f'{best_feature} in [{cat_label}]'
    else:
        node.threshold = best_threshold
        left_idx = [i for i, r in enumerate(records) if r[best_feature] <= best_threshold]
        right_idx = [i for i, r in enumerate(records) if r[best_feature] > best_threshold]

        if best_feature == 'hour':
            node.reason = f'hour <= {best_threshold:.0f} IST'
        elif best_feature == 'day_of_week':
            day_name = DAY_NAMES[int(best_threshold)] if best_threshold < 7 else str(best_threshold)
            node.reason = f'day <= {day_name}'
        else:
            node.reason = f'{best_feature} <= {best_threshold:.4f}'

    left_records = [records[i] for i in left_idx]
    left_labels = [labels[i] for i in left_idx]
    right_records = [records[i] for i in right_idx]
    right_labels = [labels[i] for i in right_idx]

    node.left = build_tree(left_records, left_labels, class_weights, depth + 1, max_depth, min_samples)
    node.right = build_tree(right_records, right_labels, class_weights, depth + 1, max_depth, min_samples)

    return node


def tree_to_dict(node):
    d = {
        'samples': node.samples,
        'distribution': {ACTION_LABELS[int(k)]: v for k, v in node.distribution.items()},
        'outcomeStats': node.outcome_stats,
    }

    if node.prediction is not None:
        d['action'] = ACTION_LABELS[node.prediction]
        d['actionDescription'] = ACTION_DESCRIPTIONS[ACTION_LABELS[node.prediction]]
        d['confidence'] = round(
            node.distribution.get(node.prediction, 0) / node.samples * 100, 1
        ) if node.samples > 0 else 0
    else:
        d['splitFeature'] = node.feature
        d['reason'] = node.reason
        if node.threshold is not None:
            d['threshold'] = node.threshold
        if node.categories is not None:
            d['categories'] = node.categories
        d['yes'] = tree_to_dict(node.left)
        d['no'] = tree_to_dict(node.right)

    return d


def predict(node, record):
    if node.prediction is not None:
        return node.prediction
    val = record[node.feature]
    if node.categories is not None:
        if val in set(node.categories):
            return predict(node.left, record)
        else:
            return predict(node.right, record)
    else:
        if val <= node.threshold:
            return predict(node.left, record)
        else:
            return predict(node.right, record)


def evaluate_tree(node, records, labels):
    correct = 0
    for r, label in zip(records, labels):
        pred = predict(node, r)
        if pred == label:
            correct += 1
    return correct / len(records) if records else 0


# --- Junction & Police Station intelligence ---

def compute_junction_stats(records):
    """Compute per-junction incident stats for barricade intelligence."""
    junction_data = defaultdict(lambda: {
        'count': 0, 'high_priority': 0, 'closures': 0,
        'causes': Counter(), 'lat': 0, 'lng': 0,
        'corridors': Counter(), 'police_stations': Counter(),
        'resolution_times': [],
    })

    for r in records:
        jn = r['junction']
        if not jn:
            continue
        d = junction_data[jn]
        d['count'] += 1
        d['lat'] += r['lat']
        d['lng'] += r['lng']
        if r['priority'] == 'High':
            d['high_priority'] += 1
        if r['road_closure']:
            d['closures'] += 1
        d['causes'][r['cause']] += 1
        if r['corridor'] and r['corridor'] != 'Non-corridor':
            d['corridors'][r['corridor']] += 1
        if r['police_station']:
            d['police_stations'][r['police_station']] += 1
        if r['resolution_min'] is not None:
            d['resolution_times'].append(r['resolution_min'])

    result = {}
    for jn, d in junction_data.items():
        if d['count'] < 3:
            continue
        res_times = d['resolution_times']
        result[jn] = {
            'count': d['count'],
            'lat': round(d['lat'] / d['count'], 6),
            'lng': round(d['lng'] / d['count'], 6),
            'highPriorityRate': round(d['high_priority'] / d['count'], 2),
            'closureRate': round(d['closures'] / d['count'], 2),
            'topCauses': [{'cause': c, 'count': n} for c, n in d['causes'].most_common(3)],
            'corridors': [c for c, _ in d['corridors'].most_common(2)],
            'policeStations': [ps for ps, _ in d['police_stations'].most_common(2)],
            'medianResolutionMin': round(float(np.median(res_times)), 1) if res_times else None,
        }

    return result


def compute_police_station_stats(records):
    """Compute per-police-station deployment stats."""
    ps_data = defaultdict(lambda: {
        'count': 0, 'high_priority': 0, 'closures': 0,
        'resolution_times': [], 'causes': Counter(),
        'lat': 0, 'lng': 0, 'junctions': Counter(),
    })

    for r in records:
        ps = r['police_station']
        if not ps:
            continue
        d = ps_data[ps]
        d['count'] += 1
        d['lat'] += r['lat']
        d['lng'] += r['lng']
        if r['priority'] == 'High':
            d['high_priority'] += 1
        if r['road_closure']:
            d['closures'] += 1
        d['causes'][r['cause']] += 1
        if r['resolution_min'] is not None:
            d['resolution_times'].append(r['resolution_min'])
        if r['junction']:
            d['junctions'][r['junction']] += 1

    result = {}
    for ps, d in ps_data.items():
        res = d['resolution_times']
        result[ps] = {
            'count': d['count'],
            'lat': round(d['lat'] / d['count'], 6),
            'lng': round(d['lng'] / d['count'], 6),
            'highPriorityRate': round(d['high_priority'] / d['count'], 2),
            'medianResolutionMin': round(float(np.median(res)), 1) if res else None,
            'topJunctions': [{'name': j, 'count': n} for j, n in d['junctions'].most_common(5)],
        }

    return result


def compute_corridor_cooccurrence(records):
    """Find which corridors have simultaneous incidents (same day).
    High co-occurrence = don't divert to that corridor."""
    daily_corridors = defaultdict(set)
    for r in records:
        if r['corridor'] and r['corridor'] != 'Non-corridor':
            daily_corridors[r['date_key']].add(r['corridor'])

    cooccurrence = defaultdict(Counter)
    for day, corridors in daily_corridors.items():
        for c in corridors:
            for other in corridors:
                if c != other:
                    cooccurrence[c][other] += 1

    # Normalize: for each corridor, find which others are LEAST
    # co-occurring (best diversion candidates)
    all_corridors = list(cooccurrence.keys())
    total_days = len(daily_corridors)

    result = {}
    for c in all_corridors:
        others = []
        for other in all_corridors:
            if other == c:
                continue
            co_days = cooccurrence[c].get(other, 0)
            co_rate = co_days / total_days
            others.append({'corridor': other, 'coRate': round(co_rate, 3), 'coDays': co_days})

        # Sort by co-occurrence rate ascending (least co-occurring = best diversion)
        others.sort(key=lambda x: x['coRate'])
        result[c] = {
            'bestDiversions': [o['corridor'] for o in others[:3]],
            'avoidDiversions': [o['corridor'] for o in others[-3:]],
            'details': others[:5],
        }

    return result


def compute_action_stats(records, labels):
    """Compute detailed stats per action class."""
    stats = {}
    for action_idx in range(4):
        action = ACTION_LABELS[action_idx]
        action_records = [records[i] for i, l in enumerate(labels) if l == action_idx]
        res_times = [r['resolution_min'] for r in action_records if r['resolution_min'] is not None]
        corridors = Counter(r['corridor'] for r in action_records if r['corridor'] != 'Non-corridor')
        junctions = Counter(r['junction'] for r in action_records if r['junction'])
        police_stations = Counter(r['police_station'] for r in action_records if r['police_station'])
        veh_types = Counter(r['veh_type'] for r in action_records if r['veh_type'])

        stats[action] = {
            'count': len(action_records),
            'medianResolutionMin': round(float(np.median(res_times)), 1) if res_times else None,
            'avgResolutionMin': round(float(np.mean(res_times)), 1) if res_times else None,
            'topCorridors': [{'name': c, 'count': n} for c, n in corridors.most_common(5)],
            'topJunctions': [{'name': j, 'count': n} for j, n in junctions.most_common(5)],
            'topPoliceStations': [{'name': ps, 'count': n} for ps, n in police_stations.most_common(5)],
            'vehicleTypes': [{'type': v, 'count': n} for v, n in veh_types.most_common(5)],
        }
    return stats


# --- Main ---

def main():
    print('Loading data...')
    records = load_dataset()
    labels = [r['label'] for r in records]
    print(f'  {len(records)} records loaded')
    print(f'  Action distribution: {Counter(labels)}')

    # Compute inverse-frequency class weights
    total = len(labels)
    class_counts = Counter(labels)
    n_classes = len(class_counts)
    class_weights = {}
    for cls, count in class_counts.items():
        class_weights[cls] = total / (n_classes * count)
    print(f'  Class weights: {class_weights}')

    # Chronological split
    split = int(0.8 * len(records))
    train_rec, test_rec = records[:split], records[split:]
    train_lab, test_lab = labels[:split], labels[split:]

    print(f'\nTrain: {len(train_rec)}, Test: {len(test_rec)}')

    print('\nBuilding decision tree (max_depth=7, class-weighted)...')
    tree = build_tree(train_rec, train_lab, class_weights, max_depth=7, min_samples=8)

    train_acc = evaluate_tree(tree, train_rec, train_lab)
    test_acc = evaluate_tree(tree, test_rec, test_lab)
    print(f'  Train accuracy: {train_acc:.3f}')
    print(f'  Test accuracy:  {test_acc:.3f}')

    for action_idx in range(4):
        action_test = [(r, l) for r, l in zip(test_rec, test_lab) if l == action_idx]
        if action_test:
            correct = sum(1 for r, l in action_test if predict(tree, r) == l)
            print(f'    {ACTION_LABELS[action_idx]}: {correct}/{len(action_test)} '
                  f'({correct/len(action_test)*100:.1f}%)')

    # Compute intelligence layers
    print('\nComputing junction intelligence...')
    junction_stats = compute_junction_stats(records)
    print(f'  {len(junction_stats)} junctions with 3+ incidents')

    print('Computing police station intelligence...')
    ps_stats = compute_police_station_stats(records)
    print(f'  {len(ps_stats)} police stations')

    print('Computing corridor co-occurrence...')
    corridor_cooccurrence = compute_corridor_cooccurrence(records)
    print(f'  {len(corridor_cooccurrence)} corridors analyzed')

    print('Computing action stats...')
    action_stats = compute_action_stats(records, labels)

    # Export
    print('\nExporting...')
    output = {
        'tree': tree_to_dict(tree),
        'actionStats': action_stats,
        'actionLabels': ACTION_LABELS,
        'actionDescriptions': ACTION_DESCRIPTIONS,
        'junctionIntelligence': junction_stats,
        'policeStationIntelligence': ps_stats,
        'corridorDiversions': corridor_cooccurrence,
        'metrics': {
            'trainAccuracy': round(train_acc, 3),
            'testAccuracy': round(test_acc, 3),
            'trainSize': len(train_rec),
            'testSize': len(test_rec),
            'classWeights': {ACTION_LABELS[k]: round(v, 2) for k, v in class_weights.items()},
        },
    }

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)

    size_kb = os.path.getsize(OUTPUT_JSON) / 1024
    print(f'Exported to {OUTPUT_JSON} ({size_kb:.1f} KB)')


if __name__ == '__main__':
    main()
