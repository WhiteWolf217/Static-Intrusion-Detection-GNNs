from scapy.all import rdpcap
import pandas as pd
from tqdm import tqdm
def pcap_to_flows(pcap_path, timeout=120):
    packets=rdpcap(pcap_path)
    flows={}
    for pkt in tqdm(packets,desc='Processinh'):
        if not hasattr(pkt,'ip'): continue

        src=f"{pkt.ip.src}:{pkt.sport if hasattr(pkt,'sport') else 0}"
        dst=f"{pkt.ip.dst}:{pkt.dport if hasattr(pkt,'dport') else 0}"
        flow_key=tuple(sorted((src,dst)))

        if flow_key not in flows:
            flows[flow_key]={
                'start_time': pkt.time,
                'end_time': pkt.time,
                'fwd_packets': 0,
                'bwd_packets': 0,
                'fwd_bytes': 0,
                'bwd_bytes': 0,
                'src_ip': pkt.ip.src,
                'dst_ip': pkt.ip.dst,
                'src_port': pkt.sport if hasattr(pkt, 'sport') else 0,
                'dst_port': pkt.dport if hasattr(pkt, 'dport') else 0
            }
        flow=flows[flow_key]
        direction='fwd' if (src,flow['src_port'])==(pkt.ip.src,flow['src_port']) else 'bwd'
        flow[f'{direction}_packets'] +=1
        flow[f'{direction}_bytes'] +=len(pkt)
        flow['end_time']=pkt.time

    return pd.DataFrame([
            {
                'Source IP': flow['src_ip'],
                'Destination IP': flow['dst_ip'],
                'Source Port': flow['src_port'],
                'Destination Port': flow['dst_port'],
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

