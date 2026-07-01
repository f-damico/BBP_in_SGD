import os, time
import numpy as np

import argparse

def identity(x):
  return x

def identityp(x):
  return np.ones_like(x)

def identitypp(x):
  return np.zeros_like(x)

def tanhp(x):
  return np.ones_like(x) - np.tanh(x)**2

def tanhpp(x):
  return 2.*np.tanh(x)**3 - 2.*np.tanh(x)

def relu(x):
  return x*np.heaviside(x, 0)

def relup(x):
  return np.heaviside(x, 0)

def relupp(x):
  eps = 1e-15 * np.ones_like(x)
  return np.where(np.abs(x) > eps, 0., 1./eps)

def simulate_time(epsilon, sigma, N, M, n_try, n_data, f, fp, data_dir, epochs, W_star, alpha, u_star):
  r = N / M

  # Load or generate data
  data_string = "dyn_N%d_r%.2f_eps%.5f_sig%.5f_alpha%d"%(N, r, epsilon, sigma, alpha)
  print(data_string)

  README = ""

  print("=====")
  print("Generating ... ")
  u_top_list = np.zeros((epochs+1, n_try, N))
  q2_list = np.zeros((epochs+1, n_try))

  data = np.random.normal(0., 1., size=(n_try, n_data, M))
  W_ = np.random.normal(0., sigma/np.sqrt(M), size=(n_try, N, M))
  
  h_hat = np.einsum("ndM, nNM -> ndN", data, W_) # n_try x n_data x N
  y_hat = f(h_hat) # n_try x n_data x N
  y = f(np.einsum("ndM, nNM -> ndN", data, W_star)) # n_try x n_data x N

  delta = (y - y_hat) * fp(h_hat) # n_try x n_data x N
  dl = - (np.einsum("ndN, ndM -> nNM", delta, data)) / n_data # n_try x N x M
  
  # For linear
  #dl = - (W_star - W_)
  
  # Single Step Update
  W_ -= epsilon * dl
  Xp = np.einsum("nac, nbc -> nab", W_, W_)

  # From here
  _, u_ = np.linalg.eigh(Xp)
  u_top_list[0] = u_[:, :, -1].copy()

  q2_ = np.sum(u_[:, :, -1] * u_star, axis=-1)**2
  q2_list[0] = q2_.copy()
  
  for e in range(epochs):
    for _ in range(25):
      h_hat = np.einsum("ndM, nNM -> ndN", data, W_) # n_try x n_data x N
      y_hat = f(h_hat) # n_try x n_data x N
      y = f(np.einsum("ndM, nNM -> ndN", data, W_star)) # n_try x n_data x N

      delta = (y - y_hat) * fp(h_hat) # n_try x n_data x N
      dl = - (np.einsum("ndN, ndM -> nNM", delta, data)) / n_data # n_try x N x M

      # Single Step Update
      W_ -= epsilon * dl
    
    Xp = np.einsum("nac, nbc -> nab", W_, W_)
    
    try:
      _, u_ = np.linalg.eigh(Xp)
    except:
      u_ = np.zeros_like(Xp)

    u_top_list[e+1] = u_[:, :, -1].copy()

    q2_ = np.sum(u_[:, :, -1] * u_star, axis=-1)**2
    q2_list[e+1] = q2_.copy()

  # Generate the docstring
  README = """
  N: %d
  M: %d
  r: %.4f
  eps: %.4f
  sig: %.4f
  n_try: %d
  alpha: %d
  """%(N, M, r, epsilon, sigma, n_try, alpha)

  np.savez(data_dir+data_string+"_data.npz",
           README = README,
           q2_list = q2_list, 
           u_top_list = u_top_list,
           u_star = u_star
          )

  print("Data saved to "+data_dir+data_string+"_data.npz")
  print(README)
  print("=====")

def main(args):
  N = 100
  M = 200
  alpha = 200
  r = N / M
  
  n_try = 15
  n_data = M * alpha
  epochs = 8

  u = np.random.normal(0., 1., size=(N))
  u /= np.linalg.norm(u)
  print(u)

  v = np.random.normal(0., 1., size=(M))
  v /= np.linalg.norm(v)
  print(v)
  
  ## u = e
  #u = np.zeros((N))
  #u[0] = 1
  #v = np.zeros((M))
  #v[0] = 1

  W_star = np.einsum("ni, nj -> nij", [u]*n_try, [v]*n_try)

  if args.actf == "identity":
    f = identity
    fp = identityp
    fpp = identitypp
  elif args.actf == "relu":
    f = relu
    fp = relup
    fpp = relupp
  elif args.actf == "tanh":
    f = np.tanh
    fp = tanhp
    fpp = tanhpp
  else:
    raise ValueError("Undefined activation")

  simulate_time(
      args.epsilon, 
      args.sigma, 
      N=N, M=M, 
      n_try=n_try, 
      n_data=n_data, 
      f=f, fp=fp, 
      data_dir=args.data_dir, 
      epochs=epochs, 
      W_star=W_star, alpha=alpha, u_star=u
      )
  
  print("done")
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description="BBP")
  parser.add_argument("--data_dir", type=str, default="./")
  parser.add_argument("--actf", type=str, default="identity")
  parser.add_argument("--epsilon", type=float, default=1.)
  parser.add_argument("--sigma", type=float, default=1.)
  args = parser.parse_args()

  main(args)
