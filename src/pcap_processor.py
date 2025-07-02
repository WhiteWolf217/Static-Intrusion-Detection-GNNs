from scapy.all import rdpcap
import pandas as pd
from tqdm import tqdm
from scapy.layers.inet import IP
def pcap_to_flows(pcap_path, timeout=120):
    packets=rdpcap(pcap_path)
    print(f"[INFO] Loaded {len(packets)} packets from {pcap_path}")
    flows={}
    for pkt in tqdm(packets, desc='Processing'):
        if not pkt.haslayer(IP):
            #print("[SKIP] Non-IP packet")
            continue

        ip_layer = pkt[IP]
        src_port = pkt.sport if hasattr(pkt, 'sport') else 0
        dst_port = pkt.dport if hasattr(pkt, 'dport') else 0
        src = f"{ip_layer.src}:{src_port}"
        dst = f"{ip_layer.dst}:{dst_port}"
        flow_key = tuple(sorted((src, dst)))

        if flow_key not in flows:
            flows[flow_key] = {
                'start_time': pkt.time,
                'end_time': pkt.time,
                'fwd_packets': 0,
                'bwd_packets': 0,
                'fwd_bytes': 0,
                'bwd_bytes': 0,
                'src_ip': ip_layer.src,
                'dst_ip': ip_layer.dst,
                'src_port': src_port,
                'dst_port': dst_port
            }

        flow = flows[flow_key]
        direction = 'fwd' if ip_layer.src == flow['src_ip'] else 'bwd'
        flow[f'{direction}_packets'] += 1
        flow[f'{direction}_bytes'] += len(pkt)
        flow['end_time'] = pkt.time

    return pd.DataFrame([
            {
                'Source IP': flow['src_ip'],
                'Destination IP': flow['dst_ip'],
                'Source Port': str(flow['src_port']),
                'Destination Port': str(flow['dst_port']),
                'Flow Duration': flow['end_time'] - flow['start_time'],
                'Total Fwd Packets': flow['fwd_packets'],
                'Total Backward Packets': flow['bwd_packets'],
                'Total Length of Fwd Packets': flow['fwd_bytes'],
                'Total Length of Bwd Packets': flow['bwd_bytes'],
                'Flow Bytes/s': (flow['fwd_bytes'] + flow['bwd_bytes']) / max(1e-6, flow['end_time'] - flow['start_time']),
                'Flow Packets/s': (flow['fwd_packets'] + flow['bwd_packets']) / max(1e-6, flow['end_time'] - flow['start_time'])
            }
            for flow in flows.values()
        ])
