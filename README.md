# lts-coursework-experiments

Early coursework assignments and network packet analysis experiments built during **Practice School I (PS-1)** at BITS Pilani.

This is the first repository in the LTS project series. It contains the initial learning phase — implementing ML-based intrusion detection classifiers and experimenting with live packet sniffing using Scapy.

---

## How This Fits Into the Larger Work

This repo represents the **foundation phase**:

```
lts-coursework-experiments  ← you are here
        ↓
lts-monolithic-ddos-detection   (single-machine DDoS sniffer + ML inference)
        ↓
lts-distributed-ddos-detection  (distributed Kafka-based detection pipeline)

healthcare-prediction           (separate parallel project — Streamlit ML dashboard)
```

---

## What's in This Repo

### Code (top-level)
| File | Description |
|---|---|
| `assign1first.py` – `assign8perp.py` | Progressive coursework assignments: Random Forest, XGBoost, LightGBM, Ensemble classifiers on cybersecurity intrusion data |
| `SNA.py` / `SNA-trial.py` | Early Scapy-based live network packet sniffer with protocol analysis |
| `scapy_trial.py` | Scapy library exploration / playground |
| `net_scrape.py` | Basic web scraping experiment |
| `rest_api_request.py` | REST API consumption example |
| `send_email.py` | Simple email notification utility |
| `code1.py` | Minimal scratch script |
| `requirements.txt` | Python dependencies |

### Documents (`documents/`)
| Subfolder | Contents |
|---|---|
| `documents/reports/` | PS-1 assignment PDFs, midsem report, monthly progress report |
| `documents/presentations/` | Assignment 1 ML presentation slide deck |

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run any assignment:
```bash
python assign1first.py
```

Run the live packet sniffer (requires admin/root):
```bash
sudo python SNA.py
```

---

## Data Files Not Included

Raw datasets are **not** in this repo. They live locally under `raw data collected for project/`:
- `cybersecurity_intrusion_data.csv` — used by assignments 1–8
- `NTD.csv` — auxiliary network traffic data

The code resolves dataset paths dynamically relative to its location, so update the path constant at the top of each script if running from a different layout.

---

## See Also
- [`PS1-monolithic-ddos-detection`](https://github.com/nikunj-gupta-1/PS1-monolithic-ddos-detection) — monolithic single-machine DDoS detection system
- [`PS1-distributed-ddos-detection`](https://github.com/nikunj-gupta-1/PS1-distributed-ddos-detection) — distributed multi-device pipeline
- [`healthcare-prediction`](https://github.com/nikunj-gupta-1/healthcare-prediction) — separate Streamlit healthcare ML dashboard
