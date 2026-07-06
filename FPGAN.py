pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-fidelity lpips

# Imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torchvision.transforms.functional as TF
from tqdm import tqdm
import numpy as np
import random
import time
import itertools
import shutil
from torch_fidelity.metrics import calculate_metrics 
import lpips 
from PIL import Image
import json
import sys
import matplotlib.pyplot as plt

# --- Dataset Download (KaggleHub) ---
try:
    import kagglehub
    HAS_KAGGLEHUB = True
except ImportError:
    HAS_KAGGLEHUB = False
    print("Warning: 'kagglehub' not found. Install via `pip install kagglehub` for auto-download.")

# IPython display check
try:
    from IPython.display import display, FileLink
    IS_NOTEBOOK = True
except ImportError:
    IS_NOTEBOOK = False

# --- Configuration ---
IMG_SIZE = 128
LATENT_DIM = 100
BATCH_SIZE = 32  # Consistent with paper reporting
TOTAL_ITERATIONS = 25000
METRIC_INTERVAL = 1500
NUM_EVAL_IMAGES = 5000
NUM_LPIPS_PAIRS = 5000
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IS_GPU = torch.cuda.is_available()
NUM_WORKERS = 4 
SAVE_PERIODIC_CHECKPOINTS = False

# --- Hyperparameters ---
LR_G = 0.00022
LR_D = 0.0002
BETA1 = 0.5
BETA2 = 0.999

# --- Sliding Window Config ---
KDE_BANDWIDTH = 0.5
WINDOW_RADIUS = 2  # Window: 2 left + 2 right

# --- Ablation Flags ---
FREEZE_PERCEPTRON = True        

# --- Dataset Setup ---
DATASET_SLUG = "orionrid/animals-256x256"

def setup_dataset():
    """
    Downloads dataset using kagglehub and resolves the image directory.
    """
    # 1. Check if we are in a Kaggle Kernel with pre-mounted data
    kaggle_mount = "/kaggle/input/animals-256x256"
    if os.path.exists(kaggle_mount):
        print(f"Found dataset in Kaggle input: {kaggle_mount}")
        path = kaggle_mount
    
    # 2. Use KaggleHub to download
    elif HAS_KAGGLEHUB:
        print(f"Downloading {DATASET_SLUG} via kagglehub...")
        try:
            path = kagglehub.dataset_download(DATASET_SLUG)
            print(f"Dataset downloaded to: {path}")
        except Exception as e:
            print(f"KaggleHub download failed: {e}")
            sys.exit(1)
    else:
        print("Error: Dataset not found and 'kagglehub' library is missing.")
        print("Please install it or manually set the data path.")
        sys.exit(1)

    # 3. Dynamic Path Resolution
    # kagglehub might return a root folder. We need the folder containing actual images.
    print("Locating image directory...")
    for root, dirs, files in os.walk(path):
        if any(f.lower().endswith(('.jpg', '.png')) for f in files):
            return root
            
    return path

# Resolve Data Path
try:
    TRAIN_DIR = setup_dataset()
    EVAL_REAL_DIR = TRAIN_DIR
    print(f"Training Data Path: {TRAIN_DIR}")
except Exception as e:
    print(f"Critical Error setting up dataset: {e}")
    sys.exit(1)

# --- Output Directories ---
config_name = f"fp_SlidingWindowKDE_R{WINDOW_RADIUS}"
OUTPUT_IMAGE_DIR = f"generated_images_{config_name}_128"
CHECKPOINT_DIR = f"checkpoints_{config_name}_128"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# --- Utility Functions ---
VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if IS_GPU:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {seed}")

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

# --- Custom Dataset ---
class FlatImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir))
                            if os.path.isfile(os.path.join(root_dir, f)) and
                            os.path.splitext(f)[1].lower() in VALID_EXTENSIONS]
        if not self.image_files:
            raise FileNotFoundError(f"No valid images found in {root_dir}")

    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_files[idx]).convert('RGB')
        except:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color='black')
        if self.transform: img = self.transform(img)
        return img, 0

# --- Architectures ---
ngf, ndf = 64, 64

class Generator(nn.Module):
    def __init__(self, nz=LATENT_DIM, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*16, 4, 1, 0, bias=False), nn.BatchNorm2d(ngf*16), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*16, ngf*8, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf*8), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf*4), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf*2), nn.ReLU(True),
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False), nn.BatchNorm2d(ngf), nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Tanh()
        )
    def forward(self, input): return self.main(input.view(input.size(0), input.size(1), 1, 1))

class Discriminator(nn.Module):
    def __init__(self, nc=3):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*8, ndf*16, 4, 2, 1, bias=False), nn.BatchNorm2d(ndf*16), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf*16, 1, 4, 1, 0, bias=False) 
        )
    def forward(self, input): return self.main(input)

class ParityNet(nn.Module):
    def __init__(self):
        super(ParityNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(1, 10), nn.ReLU(),
            nn.Linear(10, 1), 
            nn.Sigmoid() 
        )
    def forward(self, x):
        return self.main(torch.abs(x).view(x.size(0), -1))

# --- Metrics ---
try: lpips_fn = lpips.LPIPS(net='alex').to(DEVICE)
except: lpips_fn = None

def compute_metrics(netG, eval_dir, latent_dim, n_eval, batch_size, device, is_gpu):
    netG.eval()
    fake_dir = f"./temp_fake_{config_name}"
    os.makedirs(fake_dir, exist_ok=True)
    fake_tensors = []
    with torch.no_grad():
        for i in tqdm(range(0, n_eval, batch_size)):
            b_sz = min(batch_size, n_eval - i)
            z = torch.randn(b_sz, latent_dim, device=device)
            fakes = netG(z).cpu()
            for j, fake in enumerate(fakes):
                vutils.save_image(fake, f"{fake_dir}/{i+j:05d}.png", normalize=True)
            fake_tensors.append(fakes)
    fake_full = torch.cat(fake_tensors)

    try:
        metrics = calculate_metrics(input1=fake_dir, input2=eval_dir, cuda=is_gpu, 
                                    isc=True, fid=True, kid=True, verbose=False,
                                    input2_cache_name=f"cache_{os.path.basename(EVAL_REAL_DIR)}")
    except Exception as e:
        print(f"Metric error: {e}")
        metrics = {'frechet_inception_distance': float('inf')}

    if lpips_fn:
        dists = []
        for _ in tqdm(range(0, NUM_LPIPS_PAIRS, batch_size)):
            idx1 = torch.randint(0, n_eval, (batch_size,))
            idx2 = torch.randint(0, n_eval, (batch_size,))
            img1 = torch.clamp(fake_full[idx1], -1, 1).to(device)
            img2 = torch.clamp(fake_full[idx2], -1, 1).to(device)
            dists.extend(lpips_fn(img1, img2).cpu().detach().numpy().flatten())
        metrics['lpips_diversity'] = np.mean(dists)

    if os.path.exists(fake_dir): shutil.rmtree(fake_dir)
    netG.train()
    return metrics

# --- Sliding Window KDE Loss ---
def calculate_sliding_window_kde(fake_scores, real_scores, bandwidth, window_radius=2):
    # 1. Sort Real Scores
    real_sorted, _ = torch.sort(real_scores.detach(), dim=0)
    N_real = len(real_sorted)
    
    # 2. Find Center Indices via Binary Search
    center_indices = torch.searchsorted(real_sorted.squeeze(), fake_scores.squeeze())
    
    # 3. Gather Window Neighbors
    kernel_sum = 0
    sigma_sq = bandwidth ** 2
    
    for offset in range(-window_radius, window_radius):
        target_idx = torch.clamp(center_indices + offset, min=0, max=N_real-1)
        neighbor_vals = torch.gather(real_sorted, 0, target_idx.unsqueeze(1))
        
        # Calculate Gaussian contribution of this neighbor
        dist_sq = (fake_scores - neighbor_vals) ** 2
        kernel_sum += torch.exp(-0.5 * dist_sq / sigma_sq)
    
    # 4. Final Density
    densities = kernel_sum + 1e-8
    loss = -torch.log(densities).mean()
    
    return loss

# --- Main Training ---
if __name__ == "__main__":
    set_seed(SEED)
    
    # Data Augmentation & Loading
    transforms_list = [
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,)*(3), (0.5,)*(3))
    ]
    
    try:
        loader = DataLoader(FlatImageFolder(TRAIN_DIR, transforms.Compose(transforms_list)),
                            batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True)
    except FileNotFoundError as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

    netG, netD, netP = Generator().to(DEVICE), Discriminator().to(DEVICE), ParityNet().to(DEVICE)
    netG.apply(weights_init); netD.apply(weights_init); netP.apply(weights_init)
    
    optG = optim.Adam(netG.parameters(), lr=LR_G, betas=(BETA1, BETA2))
    optD = optim.Adam(netD.parameters(), lr=LR_D, betas=(BETA1, BETA2))
    
    if not FREEZE_PERCEPTRON:
        optP = optim.Adam(netP.parameters(), lr=0.0002, betas=(BETA1, BETA2))
    else:
        optP = None
    
    criterion_mse = nn.MSELoss()

    best_fid = float('inf')
    metrics_history = []
    iteration_times = []
    
    print(f"Starting {TOTAL_ITERATIONS} iters of SLIDING WINDOW FP-GAN...")
    print(f"Strategy: Single-Pass + Local KDE (Radius={WINDOW_RADIUS})")
    
    start_total_time = time.time()
    start_event = torch.cuda.Event(enable_timing=True) if IS_GPU else None
    end_event = torch.cuda.Event(enable_timing=True) if IS_GPU else None
    
    data_iter = itertools.cycle(loader)

    for iter in tqdm(range(TOTAL_ITERATIONS)):
        if IS_GPU: start_event.record()
        else: iter_start = time.time()

        netG.train(); netD.train()
        if FREEZE_PERCEPTRON: netP.eval()
        else: netP.train()
        
        # --- 1. Train D ---
        try:
            real_batch = next(data_iter)[0].to(DEVICE)
        except StopIteration:
            data_iter = itertools.cycle(loader)
            real_batch = next(data_iter)[0].to(DEVICE)

        optD.zero_grad()
        if optP: optP.zero_grad()
        
        noise = torch.randn(BATCH_SIZE, LATENT_DIM, device=DEVICE)
        fake_batch = netG(noise)
        
        r_score = netD(real_batch)
        r_par = netP(r_score)
        
        f_score_d = netD(fake_batch.detach()) 
        f_par_d = netP(f_score_d)
        
        errD_real = criterion_mse(r_par, torch.ones_like(r_par))
        errD_fake = criterion_mse(f_par_d, torch.zeros_like(f_par_d))
        errD_total = errD_real + errD_fake
        
        errD_total.backward()
        optD.step()
        if optP: optP.step()
        
        # --- 2. Train G ---
        optG.zero_grad()
        
        f_score_g = netD(fake_batch)
        f_par_g = netP(f_score_g)
        
        # G Loss: Sliding Window KDE
        errG_total = calculate_sliding_window_kde(f_par_g, r_par, bandwidth=KDE_BANDWIDTH, window_radius=WINDOW_RADIUS)
        
        errG_total.backward()
        optG.step()

        # Timer End
        if IS_GPU:
            end_event.record()
            torch.cuda.synchronize()
            iteration_times.append(start_event.elapsed_time(end_event) / 1000.0)
        else:
            iteration_times.append(time.time() - iter_start)

        # --- Logging ---
        if (iter + 1) % METRIC_INTERVAL == 0 or iter == TOTAL_ITERATIONS - 1:
            m = compute_metrics(netG, EVAL_REAL_DIR, LATENT_DIM, NUM_EVAL_IMAGES, BATCH_SIZE*2, DEVICE, IS_GPU)
            m['iteration'] = iter + 1
            metrics_history.append(m)
            current_fid = m.get('frechet_inception_distance', float('inf'))
            print(f"Iter {iter+1}: FID {current_fid:.4f}")
            if current_fid < best_fid:
                best_fid = current_fid
                torch.save(netG.state_dict(), f"{CHECKPOINT_DIR}/best_G.pth")

    end_total_time = time.time()
    print(f"Finished. Total time: {(end_total_time - start_total_time)/3600:.2f}h")
