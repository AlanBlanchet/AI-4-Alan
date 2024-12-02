def dict_from_dot_keys(keys, values):
    res = {}
    for k, v in zip(keys, values):
        res_tmp = res
        levels = k.split(".")
        for level in levels[:-1]:
            res_tmp = res_tmp.setdefault(level, {})
        res_tmp[levels[-1]] = v
    return res
