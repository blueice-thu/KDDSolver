# KDDSolver

## NSL-KDD
下载链接：http://205.174.165.80/CICDataset/NSL-KDD/Dataset/NSL-KDD.zip

数据集介绍：

- https://www.unb.ca/cic/datasets/nsl.html
- [A Deeper Dive into the NSL-KDD Data Set](https://towardsdatascience.com/a-deeper-dive-into-the-nsl-kdd-data-set-15c753364657)

Statistics of redundant records in the KDD train set:

|Original records | Distinct records | Reduction rate|
|-----------------|-------------------|-----------------|
|Attacks: 3,925,650 | 262,178 | 93.32%|
|Normal: 972,781 | 812,814 | 16.44%|
|Total: 4,898,431 | 1,074,992 | 78.05%|


Statistics of redundant records in the KDD test set:


|Original records | Distinct records | Reduction rate|
|------------|---------------------|---------------|
|Attacks: 250,436 | 29,378 | 88.26%|
|Normal: 60,591 | 47,911 | 20.92%|
|Total: 311,027 | 77,289 | 75.15%|

## 分类

- Normal: 正常流量
- DoS (Denial of Service): 拒绝服务攻击
- Probe: 端口监视或扫描
- U2R (User to Root): 未授权的本地超级用户特权访问
- R2L (Remote to Local): 来自远程主机的未授权访问

|数据集|Normal|Dos|Probe|U2R|R2L|
|-----|------|---|-----|---|---|
|KDDTrain+|67343|45927|11656|52|995|
|KDDTest+|9711|7456|2421|200|2756|

### DoS

包含 11 类：apache2, back, land, mailbomb, neptune, pod, processtable, smurf, teardrop, udpstorm, worm。

### Probe

包含 6 类：ipsweep, mscan, nmap, portsweep, saint, satan。

### U2R

包含 7 类：buffer_overflow, loadmodule, perl, ps, rootkit, sqlattack, xterm。

### R2L

包含 15 类：ftp_write, guess_passwd, httptunnel, imap, multihop, named, phf, sendmail, snmpgetattack, snmpguess, spy, warezclient, warezmaster, xlock, xsnoop。
