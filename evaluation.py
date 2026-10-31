# -*- coding:utf-8 -*-

from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import (
    adjusted_mutual_info_score,
    v_measure_score,
    homogeneity_score,
)


def eval(label, pred):
    ari = ari_score(label, pred)
    nmi = nmi_score(label, pred, average_method="arithmetic")
    ami = adjusted_mutual_info_score(label, pred)
    v_measure = v_measure_score(label, pred)
    homo = homogeneity_score(label, pred)
    return ari, nmi, ami, v_measure, homo
