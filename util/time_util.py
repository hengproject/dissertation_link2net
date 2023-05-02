import time


def time_start():
    return time.perf_counter()


# return as milliseconds
def time_end(time_start):
    return round((time.perf_counter() - time_start) * 1000,3)


def time_end_as_minutes(time_start):
    return round((time.perf_counter() - time_start)  / 60,3)
