import pickle

embeddings = []
for a in range(1,5): # Pib
  for b in range(1,8): # Pix
    for c in range(1,8): # Piy
      for d in range(1,5): # Pof
        for e in [1, 3, 5, 7]: # Pkx, Pky
          for f in [1, 32, 64, 128, 256, 512]: # batch size
            for g in range(0, 25, 2): # activation buffer size in MB
              for h in range(0, 25, 2): # weight buffer size in MB
                for i in [1, 2, 3]: # mem_type
                  for j in [1, 2, 3, 4, 5, 6]: # mem_config
                    if i == 2 and j > 4:
                      continue
                    elif i == 3 and j > 1:
                      continue
                    else:
                      if g == 0:
                        g = 1
                      if h == 0:
                        h = 1
                      m = round(a*b*c*d*e*e*16*2/1024/1024/8, 3) # mask buffer size in MB
                      if m < 0.001:
                        m = 0.001
                      embeddings.append(list((a, b, c, d, e, e, f, g, h, m, i, j)))

with open('small_embeddings.pkl', 'wb') as f:
  pickle.dump(embeddings, f)
