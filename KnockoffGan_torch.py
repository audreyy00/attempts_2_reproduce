import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import argparse
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

def sample_Z(m, n, x_name):
    if ((x_name == 'Normal') | (x_name == 'AR_Normal')):
        return np.random.normal(0., np.sqrt(1./3000), size = [m, n]).copy()
    elif ((x_name == 'Uniform') | (x_name == 'AR_Uniform')):
        return np.random.uniform(-3*np.sqrt(1./3000),3*np.sqrt(1./3000),[m,n]).copy()

def sample_X(m, n):
    return np.random.permutation(m)[:n].copy()

def Permute (x):
    n = len(x[:,0])
    idx = np.random.permutation(n)
    out = x[idx,:].copy()
    return out

def sample_SH(m, n, p):
    return np.random.binomial(1, p, [m,n]).copy()

class Generator(nn.Module):
    def __init__(self, x_dim, z_dim, h_dim):
        super().__init__()
        self.fc1=nn.Linear(x_dim+z_dim,h_dim)
        self.fc2=nn.Linear(h_dim,x_dim)
        self.apply(xavier_init)

    def forward(self, x, z):
        inp = torch.cat([x, z], dim=1)
        h = torch.tanh(self.fc1(inp))
        return self.fc2(h)

class WGANDiscriminator(nn.Module):
    def __init__(self, x_dim, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(x_dim, h_dim)
        self.fc2 = nn.Linear(h_dim, 1)
        self.apply(xavier_init)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc2(h)

class Discriminator(nn.Module):
    def __init__(self, x_dim, h_dim):
        super().__init__()
        self.fc1 = nn.Linear(x_dim * 3, h_dim)
        self.fc2 = nn.Linear(h_dim, x_dim)
        self.apply(xavier_init)

    def forward(self, swap_a, swap_b, hint):
        inp = torch.cat([swap_a, swap_b, hint], dim=1)
        h = torch.tanh(self.fc1(inp))
        return torch.sigmoid(self.fc2(h))

class MINE(nn.Module):
    def __init__(self, x_dim):
        super().__init__()
        self.W1A = nn.Parameter(torch.empty(x_dim))
        self.W1B = nn.Parameter(torch.empty(x_dim))
        self.b1  = nn.Parameter(torch.zeros(x_dim))
        self.W2A = nn.Parameter(torch.empty(x_dim))
        self.W2B = nn.Parameter(torch.empty(x_dim))
        self.b2  = nn.Parameter(torch.zeros(x_dim))
        self.W3  = nn.Parameter(torch.empty(x_dim))
        self.b3  = nn.Parameter(torch.zeros(x_dim))
        self._init_weights()

    def _init_weights(self):
        for p in [self.W1A, self.W1B, self.W2A, self.W2B, self.W3]:
            nn.init.xavier_uniform_(p.unsqueeze(0))

    def forward(self, x, x_hat):
        M_h1 = torch.tanh(self.W1A * x + self.W1B * x_hat + self.b1)
        M_h2 = torch.tanh(self.W2A * x + self.W2B * x_hat + self.b2)
        M_out = self.W3 * (M_h1 + M_h2) + self.b3
        Exp_M_out = torch.exp(M_out)
        return M_out, Exp_M_out



def compute_gradient_penalty(critic, real, fake, lam, device):
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1).to(device)
    eps = eps.expand_as(real)

    interpolated = eps * real + (1 - eps) * fake
    interpolated.requires_grad_(True)

    critic_interpolated = critic(interpolated)

    grad = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]

    grad_norm = grad.view(batch_size, -1).norm(2, dim=1)
    grad_pen = lam * ((grad_norm - 1) ** 2).mean()
    return grad_pen


def KnockoffGAN_PyTorch(x_train, x_name, lamda=1.0, mu=1, mb_size=128, niter=2000):

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    n, x_dim = x_train.shape
    z_dim = x_dim
    h_dim = x_dim

    lam = 10
    lr = 1e-4


    generator = Generator(x_dim, z_dim, h_dim).to(device)
    discriminator = Discriminator(x_dim, h_dim).to(device)
    wgan_disc = WGANDiscriminator(x_dim, h_dim).to(device)
    mine = MINE(x_dim).to(device)


    opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_WD = optim.Adam(wgan_disc.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_M = optim.Adam(mine.parameters(), lr=lr, betas=(0.5, 0.999))


    for it in tqdm(range(niter)):
        for _ in range(5):

            X_idx = sample_X(n, mb_size)
            X_mb = torch.FloatTensor(x_train[X_idx, :]).to(device)
            X_perm_mb = Permute(X_mb.cpu().numpy())
            X_perm_mb = torch.FloatTensor(X_perm_mb).to(device)
            Z_mb = torch.FloatTensor(sample_Z(mb_size, z_dim, x_name)).to(device)
            S_mb = torch.FloatTensor(sample_SH(mb_size, x_dim, 0.5)).to(device)
            H_mb = torch.FloatTensor(sample_SH(mb_size, x_dim, 0.9)).to(device)

            G_sample_d = generator(X_mb, Z_mb).detach()

            opt_WD.zero_grad()
            wd_real = wgan_disc(X_mb)
            wd_fake = wgan_disc(G_sample_d)
            grad_pen = compute_gradient_penalty(wgan_disc, X_mb, G_sample_d, lam, device)
            WD_loss = wd_fake.mean() - wd_real.mean() + grad_pen
            WD_loss.backward()
            opt_WD.step()

            opt_D.zero_grad()
            SwapA = S_mb * X_mb + (1 - S_mb) * G_sample_d
            SwapB = (1 - S_mb) * X_mb + S_mb * G_sample_d
            D_out = discriminator(SwapA, SwapB, H_mb * S_mb)
            D_loss = -(S_mb * (1 - H_mb) * torch.log(D_out + 1e-8) +
                       (1 - S_mb) * (1 - H_mb) * torch.log(1 - D_out + 1e-8)).mean()
            D_loss.backward()
            opt_D.step()

            opt_M.zero_grad()
            M_out, _ = mine(X_mb, G_sample_d)
            _, Exp_M_out = mine(X_perm_mb, G_sample_d)
            M_loss = (M_out.mean(dim=0) - torch.log(Exp_M_out.mean(dim=0))).sum()
            (-M_loss).backward()
            opt_M.step()

        opt_G.zero_grad()
        X_idx = sample_X(n, mb_size)
        X_mb = torch.FloatTensor(x_train[X_idx, :]).to(device)
        X_perm_mb = Permute(X_mb.cpu().numpy())
        X_perm_mb = torch.FloatTensor(X_perm_mb).to(device)
        Z_mb = torch.FloatTensor(sample_Z(mb_size, z_dim, x_name)).to(device)
        S_mb = torch.FloatTensor(sample_SH(mb_size, x_dim, 0.5)).to(device)
        H_mb = torch.FloatTensor(sample_SH(mb_size, x_dim, 0.0)).to(device)

        G_sample = generator(X_mb, Z_mb)

        SwapA = S_mb * X_mb + (1 - S_mb) * G_sample
        SwapB = (1 - S_mb) * X_mb + S_mb * G_sample
        D_out = discriminator(SwapA, SwapB, H_mb * S_mb)
        D_loss = -(S_mb * (1 - H_mb) * torch.log(D_out + 1e-8) +
                   (1 - S_mb) * (1 - H_mb) * torch.log(1 - D_out + 1e-8)).mean()

        wd_fake = wgan_disc(G_sample)

        M_out, _ = mine(X_mb, G_sample)
        _, Exp_M_out = mine(X_perm_mb, G_sample)
        M_loss = (M_out.mean(dim=0) - torch.log(Exp_M_out.mean(dim=0))).sum()

        G_loss = -D_loss + mu * (-wd_fake.mean()) + lamda * M_loss
        G_loss.backward()
        opt_G.step()


    generator.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(x_train).to(device)
        Z_tensor = torch.FloatTensor(sample_Z(n, z_dim, x_name)).to(device)
        X_knockoff = generator(X_tensor, Z_tensor).cpu().numpy()

    return X_knockoff

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', help='input CSV file')
    parser.add_argument('-o', help='output CSV file')
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--it', default=2000, type=int, help='number of iterations')
    parser.add_argument('--target', help='target column name')
    parser.add_argument('--xname', default='Normal', help='Sample distribution [Normal, Uniform]')
    parser.add_argument('--scale', default=1, type=int, help='whether to scale data (0/1)')
    return parser.parse_args()

if __name__ == "__main__":
    args = init_arg()
    

    df = pd.read_csv(args.i)
    niter = args.it
    use_scale = args.scale
    x_name = args.xname
    lbl = args.target

    features = list(df.columns)
    features.remove(lbl)
    

    scaler = MinMaxScaler(feature_range=(0, 1))
    x = df[features]
    
    if use_scale:
        scaler.fit(x)
        x = scaler.transform(x)
    else:
        x = x.values
    

    x_k = KnockoffGAN_PyTorch(
        x,
        x_name,
        mb_size=args.bs,
        niter=niter
    )
    
   
    df_k = pd.DataFrame(x_k, columns=features)
    df_k[lbl] = df[lbl]
    df_k.to_csv(args.o, index=False)
