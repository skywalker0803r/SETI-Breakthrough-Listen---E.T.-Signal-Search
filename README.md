# SETI-Breakthrough-Listen---E.T.-Signal-Search
SETI Breakthrough Listen - E.T. Signal Search

![](https://raw.githubusercontent.com/skywalker0803r/SETI-Breakthrough-Listen---E.T.-Signal-Search/main/DSC_4014-Edit_2.jpg)

# TODO

* 1.在知識蒸餾技術上應該把student net 加上dropout 來防止overfitting提高泛化能力
* 2.知識蒸餾技術上計算loss可以參考這篇repo : https://github.com/skywalker0803r/attention-transfer/blob/master/utils.py

```
def distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels)
    return l_kl * alpha + l_ce * (1. - alpha)
```
