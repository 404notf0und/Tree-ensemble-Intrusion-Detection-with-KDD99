import pandas as pd

def map2major5(df):
    d = {
        'normal.': 0,
        'ipsweep.': 1,
        'mscan.': 1,
        'nmap.': 1,
        'portsweep.': 1,
        'saint.': 1,
        'satan.': 1,
        'apache2.': 2,
        'back.': 2,
        'mailbomb.': 2,
        'neptune.': 2,
        'pod.': 2,
        'land.': 2,
        'processtable.': 2,
        'smurf.': 2,
        'teardrop.': 2,
        'udpstorm.': 2,
        'buffer_overflow.': 3,
        'loadmodule.': 3,
        'perl.': 3,
        'ps.': 3,
        'rootkit.': 3,
        'sqlattack.': 3,
        'xterm.': 3,
        'ftp_write.': 4,
        'guess_passwd.': 4,
        'httptunnel.': 3,  # disputation resolved
        'imap.': 4,
        'multihop.': 4,  # disputation resolved
        'named.': 4,
        'phf.': 4,
        'sendmail.': 4,
        'snmpgetattack.': 4,
        'snmpguess.': 4,
        'worm.': 4,
        'xlock.': 4,
        'xsnoop.': 4,
        'spy.': 4,
        'warezclient.': 4,
        'warezmaster.': 4  # disputation resolved
    }
    l = []
    for val in df['attack_type']:
        l.append(d[val])
    tmp_df = pd.DataFrame(l, columns=['attack_type'])
    df = df.drop('attack_type', axis=1)
    df = df.join(tmp_df)
    return df

def one_hot(df):
    #print(df["service"])
    service_one_hot = pd.get_dummies(df["service"])
    df = df.drop('service', axis=1)
    df = df.join(service_one_hot)
    # test data has this column in service, clashes with protocol_type
    # and not seen in training data, won't be learn by the model, safely delete
    if 'icmp' in df.columns:
        df = df.drop('icmp', axis=1)

    protocol_type_one_hot = pd.get_dummies(df["protocol_type"])
    df = df.drop('protocol_type', axis=1)
    df = df.join(protocol_type_one_hot)

    flag_type_one_hot = pd.get_dummies(df["flag"])
    df = df.drop('flag', axis=1)
    df = df.join(flag_type_one_hot)
    return df


def merge_sparse_feature(df):
    df.loc[(df['service'] == 'ntp_u')
           | (df['service'] == 'urh_i')
           | (df['service'] == 'tftp_u')
           | (df['service'] == 'red_i')
    , 'service'] = 'normal_service_group'

    df.loc[(df['service'] == 'pm_dump')
           | (df['service'] == 'http_2784')
           | (df['service'] == 'harvest')
           | (df['service'] == 'aol')
           | (df['service'] == 'http_8001')
    , 'service'] = 'satan_service_group'
    return df
