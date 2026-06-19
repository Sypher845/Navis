"""
preprocess.py — Converts raw ASTraM CSV into optimized JSON for the web app.

Run once: python scripts/preprocess.py
Output: public/data.json
"""

import csv
import json
import os
from datetime import datetime

INPUT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'Astram event data_anonymized - Astram event data_anonymizedb40ac87.csv'
)
OUTPUT_JSON = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    'public', 'data.json'
)

# IST offset in hours
IST_OFFSET_HOURS = 5.5


def parse_datetime(dt_str):
    """Parse the datetime string from the CSV. Returns ISO string in IST or None."""
    if not dt_str or dt_str == 'NULL':
        return None
    try:
        # Format: "2024-03-07 17:01:48.111+00"
        # Strip the timezone info and parse
        clean = dt_str.split('+')[0].split('.')[0]
        dt = datetime.strptime(clean, '%Y-%m-%d %H:%M:%S')
        # The data is in UTC, convert to IST
        from datetime import timedelta
        dt_ist = dt + timedelta(hours=IST_OFFSET_HOURS)
        return dt_ist.isoformat()
    except Exception:
        return None


def compute_duration_minutes(start_str, end_str):
    """Compute duration between two datetime strings in minutes."""
    start = parse_datetime(start_str)
    end = parse_datetime(end_str)
    if not start or not end:
        return None
    try:
        s = datetime.fromisoformat(start)
        e = datetime.fromisoformat(end)
        delta = (e - s).total_seconds() / 60
        return round(delta, 1) if delta > 0 else None
    except Exception:
        return None


def safe_float(val):
    """Convert to float, return None if invalid."""
    try:
        f = float(val)
        return f if f != 0 else None
    except (ValueError, TypeError):
        return None


def safe_str(val):
    """Return string or None for empty/NULL."""
    if not val or val == 'NULL' or val.strip() == '':
        return None
    return val.strip()


def main():
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)

    records = []
    with open(INPUT_CSV, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat = safe_float(row['latitude'])
            lng = safe_float(row['longitude'])

            # Skip rows without valid coordinates
            if lat is None or lng is None:
                continue

            start_iso = parse_datetime(row['start_datetime'])
            end_iso = parse_datetime(row['end_datetime'])
            created_iso = parse_datetime(row['created_date'])
            closed_iso = parse_datetime(row.get('closed_datetime', ''))
            resolved_iso = parse_datetime(row.get('resolved_datetime', ''))

            # Compute resolution time using closed or resolved datetime
            resolution_end = closed_iso or resolved_iso
            resolution_minutes = None
            if start_iso and resolution_end:
                try:
                    s = datetime.fromisoformat(start_iso)
                    e = datetime.fromisoformat(resolution_end)
                    delta = (e - s).total_seconds() / 60
                    resolution_minutes = round(delta, 1) if delta > 0 else None
                except Exception:
                    pass

            # Extract hour of day (IST) for temporal patterns
            hour_ist = None
            if start_iso:
                try:
                    hour_ist = datetime.fromisoformat(start_iso).hour
                except Exception:
                    pass

            # Extract day of week
            day_of_week = None
            if start_iso:
                try:
                    day_of_week = datetime.fromisoformat(start_iso).weekday()
                except Exception:
                    pass

            record = {
                'id': row['id'],
                'type': row['event_type'],           # planned | unplanned
                'cause': row['event_cause'],          # vehicle_breakdown, public_event, etc.
                'lat': lat,
                'lng': lng,
                'endLat': safe_float(row['endlatitude']),
                'endLng': safe_float(row['endlongitude']),
                'address': safe_str(row['address']),
                'roadClosure': row['requires_road_closure'] == 'TRUE',
                'start': start_iso,
                'end': end_iso,
                'status': safe_str(row['status']),
                'priority': safe_str(row['priority']),
                'desc': safe_str(row['description']),
                'corridor': safe_str(row['corridor']),
                'zone': safe_str(row['zone']),
                'junction': safe_str(row['junction']),
                'policeStation': safe_str(row['police_station']),
                'vehType': safe_str(row['veh_type']),
                'direction': safe_str(row['direction']),
                'hourIST': hour_ist,
                'dayOfWeek': day_of_week,
                'resolutionMin': resolution_minutes,
                'durationMin': compute_duration_minutes(
                    row['start_datetime'], row['end_datetime']
                ),
            }
            records.append(record)

    # Sort by start time
    records.sort(key=lambda r: r['start'] or '')

    output = {
        'meta': {
            'totalRecords': len(records),
            'dateRange': {
                'start': records[0]['start'] if records else None,
                'end': records[-1]['start'] if records else None,
            },
            'generatedAt': datetime.now().isoformat(),
        },
        'incidents': records,
    }

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, separators=(',', ':'))

    print(f'Preprocessed {len(records)} records -> {OUTPUT_JSON}')
    file_size_mb = os.path.getsize(OUTPUT_JSON) / (1024 * 1024)
    print(f'Output size: {file_size_mb:.2f} MB')


if __name__ == '__main__':
    main()
