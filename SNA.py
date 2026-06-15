#!/usr/bin/env python3
# 👆 Specifying the required python config 
import platform, importlib, subprocess, sys
import os
import urllib.request
import tempfile
 
# Creating function to install npcap if not present in system already
def install_npcap():
    installer_url = "https://npcap.com/dist/npcap-1.77.exe"
    installer_path = os.path.join(tempfile.gettempdir(), "npcap.exe")
 
    print("Npcap not found. Downloading and installing...")
    urllib.request.urlretrieve(installer_url, installer_path)
 
    subprocess.run([
        installer_path,         # Silent install not possible
        "/loopback_support=no",
        "/winpcap_mode=yes",
        "/admin_only=no"
    ])
 
 
# Creating Function to Ensure Scapy library is present.
def ensure_pkg(name):
    try:
        return importlib.import_module(name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", name])
        return importlib.import_module(name)
 
scapy = ensure_pkg("scapy")
 
from scapy.config import conf
 
# Auto-configure Scapy usage by OS
system = platform.system()                
if system == "Darwin":
    # For macOS Computers, enable libpcap and set default to en0 to get packets from default network interface.
    conf.use_pcap = True                 
    conf.iface    = "en0"
elif system == "Windows":
    # For windows, verify if npcap is availible , and install if not
    try:
        subprocess.check_output("sc query npcap", shell=True)
    except subprocess.CalledProcessError:
        install_npcap()
    conf.use_pcap = True
    conf.L2listen = conf.L3socket 
else:
    # On Linux/others, no need to force pcap for packet retrieval
    conf.use_pcap = False
 
 
 
import argparse
from collections import Counter
from scapy.all import sniff, IP, TCP, UDP, ICMP, ARP, load_layer
from scapy.layers.inet6 import IPv6, ICMPv6EchoRequest
from scapy.layers.dns import DNS
from scapy.layers.http import HTTPRequest
# Load Scapy’s TLS Decrypter to try decoding HTTPS packets data
load_layer("tls")
from scapy.layers.tls.handshake import TLSClientHello
 
 
def analyze_packet(pkt, idx, counter):
    # defaults
    src, dst = "N/A", "N/A"
    sport, dport = "N/A", "N/A"
    protocol = "Not Recognised"
    info = "No HTTP/DNS/TLS info"
 
    # --- Determine Packet Type and retieve data accordingly ---
    if IPv6 in pkt:
        src, dst = pkt[IPv6].src, pkt[IPv6].dst
        if TCP in pkt:
            protocol, sport, dport = "TCP", pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            protocol, sport, dport = "UDP", pkt[UDP].sport, pkt[UDP].dport
        elif ICMPv6EchoRequest in pkt:
            protocol = "ICMPv6"
 
    elif IP in pkt:
        src, dst = pkt[IP].src, pkt[IP].dst
        if TCP in pkt:
            protocol, sport, dport = "TCP", pkt[TCP].sport, pkt[TCP].dport
        elif UDP in pkt:
            protocol, sport, dport = "UDP", pkt[UDP].sport, pkt[UDP].dport
        elif ICMP in pkt:
            protocol = "ICMP"
 
    elif ARP in pkt:
        protocol = "ARP"
        src, dst = pkt[ARP].psrc, pkt[ARP].pdst
 
    # --- Checking for Protocol Type and Extra extractable data ---
    #  For DNS query
    if pkt.haslayer(DNS) and pkt[DNS].qd:
        dns = pkt[DNS]
        qname = dns.qd.qname.decode()
        qtype = dns.qd.qtype
        info = f"DNS query={qname} type={qtype}"
        protocol = "DNS"
 
    # For HTTP request
    elif pkt.haslayer(HTTPRequest):
        http = pkt[HTTPRequest]
        method = http.Method.decode()
        host   = http.Host.decode()
        path   = http.Path.decode()
        info = f"HTTP {method} http://{host}{path}"
        protocol = "HTTP"
 
    # Try to decrypt HTTPS request using TLS Protocol
    elif TLSClientHello in pkt:
        ch = pkt[TLSClientHello]
        for ext in ch.extensions:
            if ext.ext_type == 0:  
                for name in ext.servernames:
                    hostname = name.servername.decode()
                    info = f"TLS SNI={hostname}"
                    protocol = "TLS"
                break
 
    counter[protocol] += 1
    # Returning Data extracted from packet
    return (
        f"\nPacket {idx} Analysed:\n"
        f"  From IP: {src} Port: {sport}\n"
        f"  To   IP: {dst} Port: {dport}\n"
        f"  Protocol: [{protocol}]\n"
        f"  Info: {info}"
    )
 
def main():
    # Creating Terminal line commands to take arguments
    parser = argparse.ArgumentParser(description="Network Analyzer (IPv4/6, DNS, HTTP, TLS-SNI)")
    parser.add_argument("-c", "--count", type=int, default=10, help="number of packets to capture (default:10)")
    parser.add_argument( "-i", "--iface", help="interface to sniff on (default from conf.iface)")
    args = parser.parse_args()
 
    counter = Counter()
    pkt_idx = 0
 
    def on_packet(pkt):
        nonlocal pkt_idx
        pkt_idx += 1
        print(analyze_packet(pkt, pkt_idx, counter))
 
    iface_str = args.iface or conf.iface
    print(f"Sniffing {args.count} packets on {iface_str}…")
    sniff(
        count=args.count,
        prn=on_packet,
        iface=args.iface,
        store=0
    )
 
    # Printing Total no of Packets by type
    print("\nSummary:")
    for proto, num in counter.items():
        print(f"  {proto}: {num} packets")
    print(f"Total packets captured: {pkt_idx}")
 
 
# calling MAIN function to start program
if __name__ == "__main__":
    main()
    input("\nDone. Press Enter to close...")