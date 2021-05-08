import pandas as pd
from data_process import selected_columns, encode

names = ["id", "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment",
         "urgent", "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
         "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
         "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
         "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
         "dst_host_srv_rerror_rate", "src_ip", "src_port", "dst_ip", "dst_port", "src_ip_info", "dst_ip_info",
         "conn_end_time", "label", "coarse_type", "fine_type"]

x = pd.read_csv("./exported_data/data.csv", names=names)

if x.shape[1] != 28:
    x = x.loc[:, selected_columns]
    x.to_csv("./exported_data/data.csv", index=False)
