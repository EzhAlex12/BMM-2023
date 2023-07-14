import numpy as np
cnt_iter=[1,5,2,1,1,0,1,1]
vals, counts = np.unique(cnt_iter, return_counts=True)
print(vals[np.argmax(counts)])
print("---")