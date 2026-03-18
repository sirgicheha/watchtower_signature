from datetime import datetime
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flow timeout settings (seconds)
ACTIVE_TIMEOUT = 120   # Terminate flow after 2 minutes of activity
IDLE_TIMEOUT = 30      # Terminate flow after 30 seconds of inactivity


class Flow:
    """
    Represents a bidirectional network flow identified by a five-tuple.
    Accumulates packet metadata and computes flow-level features.
    """

    def __init__(self, five_tuple, timestamp):
        src_ip, dst_ip, src_port, dst_port, protocol = five_tuple

        # Canonical five-tuple (forward direction)
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.protocol = protocol

        # Timestamps
        self.start_time = timestamp
        self.last_seen = timestamp

        # Forward packets (src -> dst)
        self.fwd_packets = []
        self.fwd_sizes = []
        self.fwd_timestamps = []
        self.fwd_flags = []

        # Backward packets (dst -> src)
        self.bwd_packets = []
        self.bwd_sizes = []
        self.bwd_timestamps = []
        self.bwd_flags = []

    def add_packet(self, metadata):
        """
        Adds a packet to the flow in the correct direction.
        """
        five_tuple = metadata['five_tuple']
        src_ip = five_tuple[0]
        timestamp = metadata['timestamp']
        size = metadata['size']
        flags = metadata['flags']

        self.last_seen = timestamp

        if src_ip == self.src_ip:
            # Forward direction
            self.fwd_packets.append(metadata)
            self.fwd_sizes.append(size)
            self.fwd_timestamps.append(timestamp)
            if flags is not None:
                self.fwd_flags.append(int(flags))
        else:
            # Backward direction
            self.bwd_packets.append(metadata)
            self.bwd_sizes.append(size)
            self.bwd_timestamps.append(timestamp)
            if flags is not None:
                self.bwd_flags.append(int(flags))

    def compute_features(self):
        """
        Computes flow-level features aligned with CICIDS2017 feature set.
        Returns a dict of features.
        """
        duration = self.last_seen - self.start_time

        fwd_count = len(self.fwd_packets)
        bwd_count = len(self.bwd_packets)
        total_count = fwd_count + bwd_count

        fwd_bytes = sum(self.fwd_sizes)
        bwd_bytes = sum(self.bwd_sizes)

        # Inter-arrival times
        all_timestamps = sorted(self.fwd_timestamps + self.bwd_timestamps)
        iats = np.diff(all_timestamps) if len(all_timestamps) > 1 else [0]

        fwd_iats = np.diff(self.fwd_timestamps) if len(
            self.fwd_timestamps) > 1 else [0]
        bwd_iats = np.diff(self.bwd_timestamps) if len(
            self.bwd_timestamps) > 1 else [0]

        # Packet length stats
        all_sizes = self.fwd_sizes + self.bwd_sizes
        fwd_sizes = self.fwd_sizes if self.fwd_sizes else [0]
        bwd_sizes = self.bwd_sizes if self.bwd_sizes else [0]

        # TCP flags
        syn_count = sum(1 for f in self.fwd_flags if f & 0x02)
        ack_count = sum(1 for f in self.fwd_flags if f & 0x10)
        fin_count = sum(1 for f in self.fwd_flags if f & 0x01)
        rst_count = sum(1 for f in self.fwd_flags if f & 0x04)
        psh_count = sum(1 for f in self.fwd_flags if f & 0x08)
        urg_count = sum(1 for f in self.fwd_flags if f & 0x20)

        # Connection fail ratio (CFR)
        # A flow with no backward packets indicates a failed connection
        cfr = 1.0 if bwd_count == 0 else 0.0

        features = {
            # Identity
            'src_ip': self.src_ip,
            'dst_ip': self.dst_ip,
            'src_port': self.src_port,
            'dst_port': self.dst_port,
            'protocol': self.protocol,

            # Temporal
            'flow_duration': duration,
            'flow_iat_mean': float(np.mean(iats)),
            'flow_iat_std': float(np.std(iats)),
            'flow_iat_max': float(np.max(iats)),
            'flow_iat_min': float(np.min(iats)),
            'fwd_iat_mean': float(np.mean(fwd_iats)),
            'fwd_iat_std': float(np.std(fwd_iats)),
            'bwd_iat_mean': float(np.mean(bwd_iats)),
            'bwd_iat_std': float(np.std(bwd_iats)),

            # Volumetric
            'total_fwd_packets': fwd_count,
            'total_bwd_packets': bwd_count,
            'total_packets': total_count,
            'total_fwd_bytes': fwd_bytes,
            'total_bwd_bytes': bwd_bytes,
            'fwd_packet_length_mean': float(np.mean(fwd_sizes)),
            'fwd_packet_length_std': float(np.std(fwd_sizes)),
            'fwd_packet_length_max': float(np.max(fwd_sizes)),
            'fwd_packet_length_min': float(np.min(fwd_sizes)),
            'bwd_packet_length_mean': float(np.mean(bwd_sizes)),
            'bwd_packet_length_std': float(np.std(bwd_sizes)),
            'bwd_packet_length_max': float(np.max(bwd_sizes)),
            'bwd_packet_length_min': float(np.min(bwd_sizes)),
            'packet_length_mean': float(np.mean(all_sizes)),
            'packet_length_std': float(np.std(all_sizes)),

            # Connection behaviour
            'syn_flag_count': syn_count,
            'ack_flag_count': ack_count,
            'fin_flag_count': fin_count,
            'rst_flag_count': rst_count,
            'psh_flag_count': psh_count,
            'urg_flag_count': urg_count,
            'connection_fail_ratio': cfr,

            # Relational
            'fwd_bwd_packet_ratio': fwd_count / bwd_count if bwd_count > 0 else fwd_count,
            'fwd_bwd_bytes_ratio': fwd_bytes / bwd_bytes if bwd_bytes > 0 else fwd_bytes,
            'down_up_ratio': bwd_bytes / fwd_bytes if fwd_bytes > 0 else 0,
        }

        return features

    def is_expired(self, current_time):
        """
        Returns True if the flow has exceeded active or idle timeout.
        """
        if current_time - self.start_time > ACTIVE_TIMEOUT:
            return True
        if current_time - self.last_seen > IDLE_TIMEOUT:
            return True
        return False


class FlowAggregator:
    """
    Manages active flows and produces completed flow records.
    """

    def __init__(self, flow_callback=None):
        self.active_flows = {}
        self.completed_flows = []
        self.flow_callback = flow_callback  # Called when a flow completes

    def get_flow_key(self, five_tuple):
        """
        Returns a canonical flow key that is direction-independent.
        (A->B and B->A belong to the same flow)
        """
        src_ip, dst_ip, src_port, dst_port, protocol = five_tuple
        # Sort to make key direction-independent
        forward = (src_ip, dst_ip, src_port, dst_port, protocol)
        backward = (dst_ip, src_ip, dst_port, src_port, protocol)
        return min(forward, backward)

    def process_packet(self, metadata):
        """
        Processes a packet metadata dict and adds it to the correct flow.
        Expires old flows as new packets arrive.
        """
        if metadata is None:
            return

        five_tuple = metadata['five_tuple']
        timestamp = metadata['timestamp']
        flow_key = self.get_flow_key(five_tuple)

        # Check for expired flows
        self._expire_flows(timestamp)

        # Add to existing flow or create new one
        if flow_key in self.active_flows:
            self.active_flows[flow_key].add_packet(metadata)
        else:
            flow = Flow(five_tuple, timestamp)
            flow.add_packet(metadata)
            self.active_flows[flow_key] = flow

    def _expire_flows(self, current_time):
        """
        Checks all active flows and completes any that have timed out.
        """
        expired_keys = [
            key for key, flow in self.active_flows.items()
            if flow.is_expired(current_time)
        ]
        for key in expired_keys:
            self._complete_flow(key)

    def _complete_flow(self, flow_key):
        """
        Moves a flow from active to completed and computes its features.
        """
        flow = self.active_flows.pop(flow_key)
        features = flow.compute_features()
        self.completed_flows.append(features)
        if self.flow_callback:
            self.flow_callback(features)

    def flush(self):
        """
        Forces completion of all remaining active flows.
        Call this at the end of a capture session.
        """
        for key in list(self.active_flows.keys()):
            self._complete_flow(key)
        logger.info(f'Flushed {len(self.completed_flows)} completed flows')

    def get_completed_flows(self):
        return self.completed_flows


if __name__ == '__main__':
    # Quick test using packet_capture module
    import sys
    from src.capture.packet_capture import capture_from_pcap, get_packet_metadata

    if len(sys.argv) < 2:
        print('Usage: python3 flow_aggregator.py <path_to_pcap>')
        sys.exit(1)

    pcap_path = sys.argv[1]
    aggregator = FlowAggregator()

    def handle_packet(packet):
        metadata = get_packet_metadata(packet)
        aggregator.process_packet(metadata)

    capture_from_pcap(pcap_path, handle_packet)
    aggregator.flush()

    flows = aggregator.get_completed_flows()
    print(f'\nTotal flows generated: {len(flows)}')
    if flows:
        print('\nSample flow features:')
        for k, v in list(flows[0].items()):
            print(f'  {k}: {v}')
