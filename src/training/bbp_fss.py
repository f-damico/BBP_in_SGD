import os, time
import numpy as np

import argparse

def main(args):
  data_dir = args.data_dir
  N = args.N
  M = N*2
  r = 0.5

  n = 100

  u = np.random.normal(0., 1., size=(N))
  u /= np.linalg.norm(u)
  v = np.random.normal(0., 1., size=(M))
  v /= np.linalg.norm(v)

  W_star = np.einsum("i, j -> ij", u, v)

  epsilon_list = np.linspace(0.4, 0.6, 50)

  q2_list = np.zeros((len(epsilon_list), n))

  for e, eps in enumerate(epsilon_list):
    print("measuring ", e, eps)
    for i in range(n):
      W0 = np.random.normal(0., 1./np.sqrt(M), size=(N, M))

      W_prime = (1. - eps) * W0 + eps * W_star
      Xp = W_prime @ W_prime.T
      _, u_ = np.linalg.eigh(Xp)
      u_top = u_[:, -1]

      q2_ = np.sum(u_top * u)**2
      q2_list[e, i] = q2_.copy()

  README = """
  N: %d
  M: %d
  r: %.4f
  n_try: %d
  """%(N, M, r, n)

  np.savez(data_dir + "q2_fss_N%d.npz"%(N), q2_list = q2_list)
  print("Saved the data to " + data_dir + "q2_fss_N%d.npz"%(N))

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="BBP_fss")
  parser.add_argument("--data_dir", type=str, default="./")
  parser.add_argument("--N", type=int, default=100)

  args = parser.parse_args()

  main(args)
