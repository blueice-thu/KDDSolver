import pandas as pd
import numpy as np

selected_columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
                    'urgent', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
                    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
                    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
                    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
                    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']

export_names = ["id", "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
                "urgent", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
                "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
                "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
                "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "src_ip", "src_port", "dst_ip", "dst_port", "src_ip_info", "dst_ip_info",
                "conn_end_time", "label", "coarse_type", "fine_type"]

# Label encode
protocol_type_list = ['tcp', 'udp', 'icmp']
service_list = ['http', 'private', 'domain_u', 'smtp', 'ftp_data', 'other', 'eco_i', 'telnet', 'ecr_i', 'ftp', 'finger',
                'pop_3', 'auth', 'imap4', 'Z39_50', 'uucp', 'courier', 'bgp', 'iso_tsap', 'uucp_path', 'whois', 'time',
                'nnsp', 'vmnet', 'urp_i', 'domain', 'ctf', 'csnet_ns', 'supdup', 'http_443', 'discard', 'gopher',
                'daytime', 'sunrpc', 'efs', 'link', 'systat', 'exec', 'name', 'hostnames', 'mtp', 'echo', 'login',
                'klogin', 'netbios_dgm', 'ldap', 'netstat', 'netbios_ns', 'netbios_ssn', 'ssh', 'kshell', 'nntp',
                'sql_net', 'IRC', 'ntp_u', 'rje', 'remote_job', 'pop_2', 'X11', 'shell', 'printer', 'pm_dump', 'tim_i',
                'urh_i', 'red_i', 'tftp_u', 'aol', 'http_8001', 'harvest', 'http_2784', 'icmp', 'oth_i']
flag_list = ['SF', 'S0', 'REJ', 'RSTR', 'RSTO', 'S1', 'SH', 'S3', 'S2', 'OTH', 'RSTOS0', 'SHR', 'RSTRH']

protocol_type_map = dict(zip(protocol_type_list, range(len(protocol_type_list))))
service_map = dict(zip(service_list, range(len(service_list))))
flag_map = dict(zip(flag_list, range(len(flag_list))))


def encode_x(x: pd.DataFrame):
    x['protocol_type'] = x['protocol_type'].map(protocol_type_map)
    x['service'] = x['service'].map(service_map)
    x['flag'] = x['flag'].map(flag_map)

    x['duration'] = x['duration'] / 6e4
    x['src_bytes'] = np.log10(x['src_bytes'] + 1) / np.log10(1.4e9)
    x['dst_bytes'] = np.log10(x['dst_bytes'] + 1) / np.log10(1.4e9)
    x['wrong_fragment'] = x['wrong_fragment'] / 3
    x['urgent'] = x['urgent'] / 3
    x['count'] = x['count'] / 511
    x['srv_count'] = x['srv_count'] / 511
    x['dst_host_count'] = x['dst_host_count'] / 255
    x['dst_host_srv_count'] = x['dst_host_srv_count'] / 255

    return x


def generate_KDD(inFile, xFile, yFile):
    data = pd.read_csv(inFile).loc[:, selected_columns + ['class']]
    x = encode_x(data.loc[:, selected_columns])
    y = data['class'].replace('normal', '0').replace('anomaly', '1')

    x.to_csv(xFile, index=False)
    y.to_csv(yFile, index=False)


def generate_normal():
    x = pd.read_csv("dataset/NormalData/NormalData.csv", names=export_names).loc[:, selected_columns]
    x = encode_x(x)
    x.to_csv("dataset/NormalData/x.csv", index=False)

    y = pd.DataFrame([0] * x.shape[0], columns=["class"])
    y.to_csv("dataset/NormalData/y.csv", index=False)


def generate_KDD_multi_y(inFile, yFile):
    y = pd.read_csv(inFile)['class']
    y.to_csv(yFile, index=False)


if __name__ == '__main__':
    # generate_normal()
    # generate_KDD("dataset/KDDTrain+/KDDTrain+_binary.csv", "dataset/KDDTrain+/x.csv", "dataset/KDDTrain+/y.csv")
    # generate_KDD("dataset/KDDTest+/KDDTest+_binary.csv", "dataset/KDDTest+/x.csv", "dataset/KDDTest+/y.csv")
    # generate_KDD_multi_y("dataset/KDDTrain+/KDDTrain+_multi.csv", "dataset/KDDTrain+/y_multi.csv")
    # generate_KDD_multi_y("dataset/KDDTest+/KDDTest+_multi.csv", "dataset/KDDTest+/y_multi.csv")
    pass
