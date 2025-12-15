#!/usr/bin/env python3

import csv
import datetime as dt
import json
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

from flask import Flask, jsonify, render_template_string, request, send_from_directory

import board
import busio
import adafruit_bh1750
import adafruit_bmp280
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn

from infer_leaf_binary import remote_infer

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

LIGHT_CSV = DATA_DIR / "light.csv"
TEMP_CSV = DATA_DIR / "temp.csv"
MOISTURE_CSV = DATA_DIR / "moisture.csv"

PHOTO_PATH = BASE_DIR / "test.jpg"

COLLECTION_INTERVAL = 300

INFER_SERVER = os.environ.get("INFER_SERVER", "http://192.168.1.73:6000")

app = Flask(__name__)

i2c = busio.I2C(board.SCL, board.SDA)
lux_sensor = adafruit_bh1750.BH1750(i2c)
bmp280 = adafruit_bmp280.Adafruit_BMP280_I2C(i2c, address=0x76)
ads = ADS.ADS1115(i2c)
moisture_chan = AnalogIn(ads, ADS.P0)

collect_lock = threading.Lock()
last_collect_time: dt.datetime | None = None
last_infer_result: Dict | None = None
last_infer_error: str | None = None

def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def timestamp_str(ts: dt.datetime | None = None) -> str:
    if ts is None:
        ts = now_utc()
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=dt.timezone.utc)
    ts = ts.astimezone(dt.timezone.utc)
    return ts.isoformat(timespec="seconds")

def append_csv(path: Path, ts: dt.datetime, value: float) -> None:
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "value"])
        writer.writerow([timestamp_str(ts), f"{value:.6f}"])

def load_series(path: Path, days: int = 7) -> Tuple[List[str], List[float]]:
    if not path.exists():
        return [], []
    cutoff = now_utc() - dt.timedelta(days=days)
    timestamps: List[str] = []
    values: List[float] = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = dt.datetime.fromisoformat(row["timestamp"])
            except Exception:
                continue
            if ts < cutoff:
                continue
            try:
                v = float(row["value"])
            except (TypeError, ValueError):
                continue
            timestamps.append(row["timestamp"])
            values.append(v)
    return timestamps, values

def read_sensors() -> Dict[str, float]:

    lux = float(lux_sensor.lux or 0.0)
    temp = float(bmp280.temperature)
    moisture_v = float(moisture_chan.voltage)
    return {"lux": lux, "temp_c": temp, "moisture_v": moisture_v}

def capture_photo() -> bool:

    import subprocess

    try:

        cmd = ["rpicam-still", "-o", str(PHOTO_PATH), "-n"]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        if proc.returncode != 0:
            print("rpicam-still error:", proc.stderr)
            return False
        return True
    except Exception as e:
        print("Error running rpicam-still:", e)
        return False

def run_photo_inference() -> None:

    global last_infer_result, last_infer_error

    if not PHOTO_PATH.exists():
        last_infer_error = "no photo file"
        last_infer_result = None
        return

    try:
        res = remote_infer(INFER_SERVER, PHOTO_PATH)
        last_infer_result = res
        last_infer_error = None
    except Exception as e:
        last_infer_error = str(e)
        last_infer_result = None

def collect_once() -> None:

    global last_collect_time
    with collect_lock:
        ts = now_utc()
        readings = read_sensors()

        append_csv(LIGHT_CSV, ts, readings["lux"])
        append_csv(TEMP_CSV, ts, readings["temp_c"])
        append_csv(MOISTURE_CSV, ts, readings["moisture_v"])

        if capture_photo():
            run_photo_inference()

        last_collect_time = ts

def collector_loop():

    while True:
        try:
            collect_once()
        except Exception as e:
            print("Collector error:", e)

        time.sleep(COLLECTION_INTERVAL)

def light_status(current: float, history: List[float]) -> str:
    if current is None:
        return "unknown"
    if len(history) < 10:
        return "building baseline"

    sorted_vals = sorted(history)
    mid = len(sorted_vals) // 2
    if len(sorted_vals) % 2 == 0:
        median = 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])
    else:
        median = sorted_vals[mid]

    if median <= 0:
        return "unknown"

    ratio = current / median
    if ratio < 0.5:
        return "needs more light"
    elif ratio > 1.8:
        return "too much light"
    else:
        return "good amount of light"

def temp_status(current: float) -> str:
    if current is None:
        return "unknown"

    if current < 15:
        return "too cold"
    elif current > 30:
        return "too hot"
    else:
        return "good temperature"

def moisture_status(current: float, history: List[float]) -> Dict[str, str | float]:

    if current is None:
        return {"status": "unknown"}

    if len(history) < 10:
        return {"status": "building baseline"}

    min_v = min(history)
    max_v = max(history)
    if max_v <= min_v:
        return {"status": "building baseline"}

    pos = (current - min_v) / (max_v - min_v)
    pos = max(0.0, min(1.0, pos))

    if pos <= 0.15:
        status = "needs water"
    elif pos >= 0.9:
        status = "too wet"
    else:
        status = "ok"

    return {
        "status": status,
        "relative_position": round(pos, 3),
        "baseline_min": round(min_v, 3),
        "baseline_max": round(max_v, 3),
    }

def build_status_payload() -> Dict:

    light_ts, light_vals = load_series(LIGHT_CSV)
    temp_ts, temp_vals = load_series(TEMP_CSV)
    moist_ts, moist_vals = load_series(MOISTURE_CSV)

    cur_light = light_vals[-1] if light_vals else None
    cur_temp = temp_vals[-1] if temp_vals else None
    cur_moist = moist_vals[-1] if moist_vals else None

    light_hist = light_vals[:-1] if len(light_vals) > 1 else light_vals
    moist_hist = moist_vals[:-1] if len(moist_vals) > 1 else moist_vals

    light_stat = light_status(cur_light, light_hist)
    temp_stat = temp_status(cur_temp)
    moist_info = moisture_status(cur_moist, moist_hist)

    photo_health = "unknown"
    photo_confidence = None
    photo_error = None

    if last_infer_error:
        photo_error = last_infer_error
    elif last_infer_result:
        pred = last_infer_result.get("pred_class", "unknown")
        p_stress = last_infer_result.get("p_stress", None)
        photo_health = pred
        if isinstance(p_stress, (float, int)):
            if pred.lower() == "stress":
                conf = float(p_stress)
            else:
                conf = float(1.0 - p_stress)
            photo_confidence = round(conf * 100.0, 1)

    return {
        "last_update": timestamp_str(last_collect_time) if last_collect_time else None,
        "light": {
            "current": cur_light,
            "status": light_stat,
            "timestamps": light_ts,
            "values": light_vals,
        },
        "temp": {
            "current": cur_temp,
            "status": temp_stat,
            "timestamps": temp_ts,
            "values": temp_vals,
        },
        "moisture": {
            "current": cur_moist,
            "info": moist_info,
            "timestamps": moist_ts,
            "values": moist_vals,
        },
        "photo": {
            "health": photo_health,
            "confidence_percent": photo_confidence,
            "error": photo_error,
            "photo_exists": PHOTO_PATH.exists(),
        },
    }

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Plant Health Monitor</title>
    <style>
        body {
            font-family: sans-serif;
            margin: 1rem 2rem;
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .card {
            border: 1px solid #ccc;
            border-radius: 0.5rem;
            padding: 0.75rem 1rem;
        }
        .card h2 {
            margin: 0 0 0.5rem 0;
            font-size: 1.1rem;
        }
        .indicator {
            font-weight: bold;
        }
        .toolbar {
            margin-bottom: 1rem;
        }
        button {
            padding: 0.5rem 1rem;
            border-radius: 0.4rem;
            border: 1px solid #444;
            background: #eee;
            cursor: pointer;
        }
        button:disabled {
            opacity: 0.6;
            cursor: default;
        }
        .charts {
            display: grid;
            grid-template-columns: 1fr;
            gap: 1.5rem;
        }
        canvas {
            max-width: 800px;
            width: 100%;
            height: 300px;
        }
        .photo-box img {
            max-width: 100%;
            height: auto;
            border-radius: 0.4rem;
            border: 1px solid #ccc;
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <h1>Plant Health Monitor</h1>

    <div class="toolbar">
        <span id="lastUpdate">Last update: (loading...)</span>
        &nbsp;&nbsp;
        <button id="updateBtn" onclick="updateNow()">Update now</button>
        <span id="updateStatus"></span>
    </div>

    <div class="status-grid">
        <div class="card">
            <h2>Photo health</h2>
            <div class="indicator" id="photoHealth">Loading...</div>
        </div>
        <div class="card">
            <h2>Light</h2>
            <div class="indicator" id="lightStatus">Loading...</div>
            <div>Current: <span id="lightCurrent">-</span> lux</div>
        </div>
        <div class="card">
            <h2>Temperature</h2>
            <div class="indicator" id="tempStatus">Loading...</div>
            <div>Current: <span id="tempCurrent">-</span> °C</div>
        </div>
        <div class="card">
            <h2>Moisture</h2>
            <div class="indicator" id="moistStatus">Loading...</div>
            <div>Current: <span id="moistCurrent">-</span> V</div>
            <div id="moistExtra"></div>
        </div>
    </div>

    <div class="card photo-box">
        <h2>Latest photo</h2>
        <div id="photoContainer">
            <em>No photo yet.</em>
        </div>
    </div>

    <h2>Sensor history (last ~7 days)</h2>
    <div class="charts">
        <div>
            <h3>Light (lux)</h3>
            <canvas id="lightChart"></canvas>
        </div>
        <div>
            <h3>Temperature (°C)</h3>
            <canvas id="tempChart"></canvas>
        </div>
        <div>
            <h3>Moisture (V)</h3>
            <canvas id="moistChart"></canvas>
        </div>
    </div>

    <script>
        let lightChart = null;
        let tempChart = null;
        let moistChart = null;

        async function fetchStatus(endpoint) {
            const resp = await fetch(endpoint);
            if (!resp.ok) {
                throw new Error("HTTP " + resp.status);
            }
            return await resp.json();
        }

        function buildChart(ctx, label, timestamps, values) {
            return new Chart(ctx, {
                type: 'line',
                data: {
                    labels: timestamps,
                    datasets: [{
                        label: label,
                        data: values,
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            ticks: {
                                maxTicksLimit: 6
                            }
                        }
                    }
                }
            });
        }

        function updateChart(chart, timestamps, values) {
            chart.data.labels = timestamps;
            chart.data.datasets[0].data = values;
            chart.update();
        }

        function updateIndicators(data) {
            document.getElementById("lastUpdate").textContent =
                "Last update: " + (data.last_update || "none yet");

            // Photo
            let ph = "unknown";
            if (data.photo) {
                ph = data.photo.health || "unknown";
                if (data.photo.confidence_percent != null) {
                    ph = ph + " (" + data.photo.confidence_percent.toFixed(1) + "% sure)";
                }
                if (data.photo.error) {
                    ph += " [error: " + data.photo.error + "]";
                }
            }
            document.getElementById("photoHealth").textContent = ph;

            // Light
            if (data.light) {
                document.getElementById("lightStatus").textContent = data.light.status;
                if (data.light.current != null) {
                    document.getElementById("lightCurrent").textContent =
                        data.light.current.toFixed(1);
                }
            }

            // Temp
            if (data.temp) {
                document.getElementById("tempStatus").textContent = data.temp.status;
                if (data.temp.current != null) {
                    document.getElementById("tempCurrent").textContent =
                        data.temp.current.toFixed(1);
                }
            }

            // Moisture
            if (data.moisture) {
                const mi = data.moisture.info || {};
                const ms = mi.status || "unknown";
                document.getElementById("moistStatus").textContent = ms;
                if (data.moisture.current != null) {
                    document.getElementById("moistCurrent").textContent =
                        data.moisture.current.toFixed(3);
                }
                let extra = "";
                if (mi.relative_position != null) {
                    extra += "Relative level: " + mi.relative_position.toFixed(3);
                }
                if (mi.baseline_min != null && mi.baseline_max != null) {
                    extra += " (7-day V range: " + mi.baseline_min.toFixed(3) +
                             "–" + mi.baseline_max.toFixed(3) + ")";
                }
                document.getElementById("moistExtra").textContent = extra;
            }

            // Photo preview
            const photoBox = document.getElementById("photoContainer");
            if (data.photo && data.photo.photo_exists) {
                photoBox.innerHTML =
                    '<img src="/photo" alt="latest plant photo">';
            } else {
                photoBox.innerHTML = "<em>No photo yet.</em>";
            }
        }

        async function loadInitial() {
            try {
                const data = await fetchStatus("/api/status");
                updateIndicators(data);

                // Build charts
                const lc = document.getElementById("lightChart").getContext("2d");
                const tc = document.getElementById("tempChart").getContext("2d");
                const mc = document.getElementById("moistChart").getContext("2d");

                lightChart = buildChart(
                    lc,
                    "lux",
                    data.light ? data.light.timestamps : [],
                    data.light ? data.light.values : []
                );
                tempChart = buildChart(
                    tc,
                    "°C",
                    data.temp ? data.temp.timestamps : [],
                    data.temp ? data.temp.values : []
                );
                moistChart = buildChart(
                    mc,
                    "volts",
                    data.moisture ? data.moisture.timestamps : [],
                    data.moisture ? data.moisture.values : []
                );
            } catch (err) {
                console.error("Initial load error:", err);
                document.getElementById("updateStatus").textContent =
                    "Error loading status: " + err;
            }
        }

        async function refreshData() {
            try {
                const data = await fetchStatus("/api/status");
                updateIndicators(data);
                if (lightChart && data.light) {
                    updateChart(lightChart, data.light.timestamps, data.light.values);
                }
                if (tempChart && data.temp) {
                    updateChart(tempChart, data.temp.timestamps, data.temp.values);
                }
                if (moistChart && data.moisture) {
                    updateChart(moistChart, data.moisture.timestamps, data.moisture.values);
                }
            } catch (err) {
                console.error("Refresh error:", err);
                document.getElementById("updateStatus").textContent =
                    "Error refreshing: " + err;
            }
        }

        async function updateNow() {
            const btn = document.getElementById("updateBtn");
            const statusEl = document.getElementById("updateStatus");
            btn.disabled = true;
            statusEl.textContent = "Collecting...";
            try {
                const resp = await fetch("/update_now", {method: "POST"});
                if (!resp.ok) {
                    throw new Error("HTTP " + resp.status);
                }
                const data = await resp.json();
                updateIndicators(data);
                if (lightChart && data.light) {
                    updateChart(lightChart, data.light.timestamps, data.light.values);
                }
                if (tempChart && data.temp) {
                    updateChart(tempChart, data.temp.timestamps, data.temp.values);
                }
                if (moistChart && data.moisture) {
                    updateChart(moistChart, data.moisture.timestamps, data.moisture.values);
                }
                statusEl.textContent = "Updated.";
                setTimeout(() => { statusEl.textContent = ""; }, 3000);
            } catch (err) {
                console.error("Update now error:", err);
                statusEl.textContent = "Error: " + err;
            } finally {
                btn.disabled = false;
            }
        }

        window.addEventListener("load", loadInitial);
    </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/status")
def api_status():
    data = build_status_payload()
    return jsonify(data)

@app.route("/update_now", methods=["POST"])
def update_now_route():
    try:
        collect_once()
    except Exception as e:
        print("update_now error:", e)
    data = build_status_payload()
    return jsonify(data)

@app.route("/photo")
def photo():
    if not PHOTO_PATH.exists():
        return "No photo", 404
    return send_from_directory(
        directory=str(PHOTO_PATH.parent),
        path=PHOTO_PATH.name,
        mimetype="image/jpeg",
    )

def start_background_collector():
    t = threading.Thread(target=collector_loop, daemon=True)
    t.start()

if __name__ == "__main__":

    start_background_collector()

    app.run(host="0.0.0.0", port=5000)
