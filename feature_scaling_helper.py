import math


def scale(data):
    avg = _cal_avg(data)
    sd = _cal_standard_deviation(data)
    scaled_data = []
    for x in data:
        scaled_data.append((x - avg) / sd)
    return scaled_data


def _cal_avg(data):
    s = sum(data)
    return s / len(data)


# it means "Do lech chuan"
def _cal_standard_deviation(data):
    avg = _cal_avg(data)
    s = 0
    for x in data:
        s = s + (x - avg) ** 2
    sd = math.sqrt(s / (len(data) - 1))
    return sd
