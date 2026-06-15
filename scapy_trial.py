#!/usr/bin/env python3
from scapy.config import conf
# Force Scapy to use libpcap on macOS
conf.use_pcap = True
conf.iface="en0"

import argparse
from collections import Counter
from scapy.all import sniff, IP, TCP, UDP, ICMP, ARP
from scapy.layers.inet6 import IPv6, ICMPv6EchoRequest
from scapy.layers.dns import DNS, DNSQR
from scapy.layers.http import HTTPRequest

def analyze_packet(pkt, idx, counter):
    # defaults
    src, dst = "N/A", "N/A"
    sport, dport = "N/A", "N/A"
    protocol = "Not Recognised"
    info = "No HTTP/DNS information found"

    # IPv6
    if IPv6 in pkt:
        src, dst = pkt[IPv6].src, pkt[IPv6].dst
        if TCP in pkt:
            protocol, sport, dport = "TCP", pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            protocol, sport, dport = "UDP", pkt[UDP].sport, pkt[UDP].dport
        elif ICMPv6EchoRequest in pkt:
            protocol = "ICMPv6"

    # IPv4
    elif IP in pkt:
        src, dst = pkt[IP].src, pkt[IP].dst
        if TCP in pkt:
            protocol, sport, dport = "TCP", pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            protocol, sport, dport = "UDP", pkt[UDP].sport, pkt[UDP].dport
        elif ICMP in pkt:
            protocol = "ICMP"

    # ARP
    elif ARP in pkt:
        protocol = "ARP"
        src, dst = pkt[ARP].psrc, pkt[ARP].pdst

    # DNS
    if pkt.haslayer(DNS):
        dns = pkt[DNS]
        qname = dns.qd.qname.decode() if dns.qd else ""
        qtype = dns.qd.qtype if dns.qd else ""
        info = f" DNS query={qname} type={qtype}"
        protocol = "DNS"

    # HTTP request
    elif pkt.haslayer(HTTPRequest):
        http = pkt[HTTPRequest]
        method = http.Method.decode()
        host   = http.Host.decode()
        path   = http.Path.decode()
        info = f" HTTP {method} http://{host}{path}"
        protocol = "HTTP"

    counter[protocol] += 1
    return f"\nPacket {idx} Analysed:\nFrom IP:{src} Port:{sport} To IP:{dst} Port:{dport}  protocol Type:[{protocol}] HTTP/DNS:{info}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--count", type=int, default=100, help="number of packets to capture")
    parser.add_argument("-i", "--iface", help="interface to sniff on")
    args = parser.parse_args()

    counter = Counter()
    pkt_idx = 0

    def on_packet(pkt):
        nonlocal pkt_idx
        pkt_idx += 1
        print(analyze_packet(pkt, pkt_idx, counter))

    print(f"Sniffing {args.count} packets on {args.iface or 'default interface'}…")
    sniff(count=args.count, prn=on_packet, iface=args.iface, store=0)

    print("\nSummary:")
    for p, n in counter.items():
        print(f"  {p}: {n}")
    print(f"Total: {pkt_idx}")

if __name__ == "__main__":
    main()
