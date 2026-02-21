import numpy as np

total_w = 12 * 12 + 12 * 2 + 24 * 12
w_flat = np.array(
    [(((i * 1664525)) % 16384) - 8192 for i in range(total_w)], dtype=np.int64
)

Ws_flat = w_flat[0 : 12 * 12] >> 2
Wa_flat = w_flat[12 * 12 : 12 * 12 + 12 * 2]
C_flat = w_flat[12 * 12 + 12 * 2 : 12 * 12 + 12 * 2 + 24 * 12]

Ws = Ws_flat.reshape((12, 12))
print("Ws Row 0:", Ws[0])
print("Ws Row 1:", Ws[1])

print("Total sum:", np.sum(w_flat))

# Now let's calculate the correlation between row i and row i+1
corr = np.corrcoef(Ws[0], Ws[1])
print("Correlation Ws 0, 1:", corr[0, 1])
