import requests

# Sample 28-feature data from your extractor
flow_data = {
    "request_id": "job_server_1_req_001",
    "src_ip": "192.168.1.100",
    "dst_ip": "10.0.0.1",
    "timestamp": 1620000000000000,
    "protocol": 6,
    "flow_duration": 1500000.0,
    "total_fwd_packets": 10,
    "total_backward_packets": 8,
    "fwd_packet_length_max": 1500.0,
    "fwd_packet_length_min": 64.0,
    "fwd_packet_length_mean": 800.0,
    "packet_length_mean": 750.0,
    "packet_length_std": 200.0,
    "flow_bytes_per_second": 5000.0,
    "flow_packets_per_second": 12.0,
    "flow_iat_mean": 125000.0,
    "flow_iat_std": 50000.0,
    "flow_iat_max": 200000.0,
    "flow_iat_min": 10000.0,
    "fwd_iat_total": 1000000.0,
    "fwd_iat_mean": 100000.0,
    "fwd_iat_std": 30000.0,
    "fwd_iat_max": 150000.0,
    "fwd_iat_min": 50000.0,
    "bwd_iat_total": 800000.0,
    "bwd_iat_mean": 100000.0,
    "bwd_iat_std": 25000.0,
    "bwd_iat_max": 130000.0,
    "bwd_iat_min": 70000.0,
    "fwd_psh_flags": 1.0,
    "bwd_psh_flags": 1.0,
    "fwd_urg_flags": 0.0
}

# Send to ML server
response = requests.post("https://n29wxtvm-8000.inc1.devtunnels.ms/predict", json=flow_data)
result = response.json()

if result['prediction'] == 1:
    print("🚨 ATTACK DETECTED!")
    print(f"Confidence: {result['confidence']:.3f}")
else:
    print("✅ Traffic is benign")
