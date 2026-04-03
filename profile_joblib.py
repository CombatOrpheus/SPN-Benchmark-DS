from joblib import Parallel, delayed
import time


def foo(x):
    return x * x


start = time.time()
Parallel(n_jobs=1)(delayed(foo)(i) for i in range(50))
print("Joblib:", time.time() - start)

start = time.time()
[foo(i) for i in range(50)]
print("List comp:", time.time() - start)
