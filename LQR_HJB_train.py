import argparse
import os
import yaml
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch import autograd
from tqdm import tqdm
import json
import copy

# Try to import RMultKAN
try:
    from rkan import RMultKAN
except ImportError:
    raise ImportError("Could not import RMultKAN from 'rkan'. Please ensure the file exists.")

# ===========================
# 1. Arguments
# ===========================

def get_args():
    parser = argparse.ArgumentParser(description="LQR HJB Equation with RMultKAN/ARMultKAN robustness analysis")

    # --- Experiment management ---
    parser.add_argument('--name', type=str, default='LQR_ARKAN_AGU', help='Experiment name')
    parser.add_argument('--model_dir', type=str, default='./models', help='Root directory for saving models')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')
    parser.add_argument('--load_ckpt', type=str, default=None, help='Path to checkpoint to resume. Disabled in robustness mode.')

    # --- PDE parameters (LQR Problem) ---
    parser.add_argument('--T', type=float, default=3.0, help='Terminal time')
    parser.add_argument('--range_min', type=float, default=-5.0, help='Spatial range min')
    parser.add_argument('--range_max', type=float, default=5.0, help='Spatial range max')

    # --- Sampling ---
    parser.add_argument('--np_i', type=int, default=21, help='Interior points per dimension')
    parser.add_argument('--np_b', type=int, default=441, help='Terminal condition points count')
    parser.add_argument('--sampling_mode', type=str, default='random', choices=['mesh', 'random'], help='Sampling mode')

    # --- Model structure (KAN) ---
    parser.add_argument('--width', type=json.loads, default=[2,6,[0,2]], help='Layer width')
    parser.add_argument('--grid', type=int, default=5, help='Initial grid size')
    parser.add_argument('--k', type=int, default=3, help='Spline order')
    parser.add_argument('--grid_eps', type=float, default=0.02, help='Init Grid Eps')

    # --- Training ---
    parser.add_argument('--steps', type=int, default=30000, help='Number of training steps')
    parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate')
    parser.add_argument('--eta_min', type=float, default=1e-5, help='Minimum learning rate for scheduler')
    parser.add_argument('--pde_weight', type=float, default=1.0, help='Weight for PDE loss')
    parser.add_argument('--bc_weight', type=float, default=1.0, help='Weight for terminal condition loss')

    # --- AGU (Adaptive Grid Update) ---
    parser.add_argument('--use_agu', action='store_true', help='Enable Adaptive Grid Update')
    parser.add_argument('--use_agu_eps', action='store_true', help='Enable Adaptive Grid Eps')
    parser.add_argument('--agu_start', type=int, default=2000, help='Step to start AGU')
    parser.add_argument('--agu_stop', type=int, default=18000, help='Step to stop AGU')
    parser.add_argument('--agu_freq', type=int, default=2000, help='Frequency of AGU execution')
    parser.add_argument('--agu_ratio', type=float, default=0.15, help='Ratio of adaptive sampling')
    parser.add_argument('--loss_window_size', type=int, default=10, help='Window size of loss history')

    parser.add_argument('--log_freq', type=int, default=1, help='Logging frequency')

    # --- Robustness analysis ---
    parser.add_argument('--robustness', action='store_true', help='Run robustness analysis over several terminal-noise levels')
    parser.add_argument('--noise_levels', type=str, default='0,0.01,0.03,0.05',
                        help='Comma-separated noise levels, e.g., 0,0.01,0.03,0.05')
    parser.add_argument('--noise_on', type=str, default='both', choices=['V', 'u', 'both'],
                        help='Which terminal target to perturb: V, u, or both')
    parser.add_argument('--noise_seed_offset', type=int, default=10000,
                        help='Offset for noise generator seed to make noise reproducible')

    return parser.parse_args()


def parse_noise_levels(noise_levels: str):
    return [float(v.strip()) for v in noise_levels.split(',') if v.strip() != '']


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


# ===========================
# 2. Physical problem definition
# ===========================

class LQR_HJB:
    def __init__(self, T, device):
        self.T = T
        self.device = device

    def exact_solution(self, x_in):
        """
        Input x_in: [t, x_state], shape (N, 2)
        Output: V(t,x), u(t,x)
        """
        t = x_in[:, 0:1]
        x = x_in[:, 1:2]

        # V(t,x) = x^2 / (1 + 2(T-t))
        denom = 1 + 2 * (self.T - t)
        V = (x ** 2) / denom

        # u(t,x) = -2x / (1 + 2(T-t))
        u = -2 * x / denom

        return V, u


# ===========================
# 3. Helpers
# ===========================

def gradients(outputs, inputs):
    grad = autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        allow_unused=True,
    )[0]
    if grad is None:
        raise RuntimeError('Gradient is None. Please check whether inputs require gradients and are used by the model.')
    return grad


def build_experiment_dir(args, noise_level=None):
    base_dir = os.path.join(args.model_dir, args.name)
    if noise_level is None:
        return base_dir
    return os.path.join(base_dir, f"noise_{int(round(noise_level * 100)):02d}pct")


def format_sci(x):
    return f"{x:.3e}"


def write_latex_table(summary_csv, tex_path, method_name='ARMultKAN-PINN'):
    rows = []
    with open(summary_csv, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    headers = []
    values = []
    for row in rows:
        nl = float(row['Noise_Level'])
        headers.append(f"{int(round(nl * 100))}\\% noise")
        values.append(format_sci(float(row['Best_L2_Error'])))

    table = []
    table.append(r"\begin{table}[t]")
    table.append(r"\centering")
    table.append(r"\caption{Robustness comparison under noisy terminal constraints.}")
    table.append(r"\label{tab:robustness_noise}")
    table.append(r"\begin{tabular}{c|" + "c" * len(headers) + r"}")
    table.append(r"\toprule")
    table.append("Method & " + " & ".join(headers) + r" \\")
    table.append(r"\midrule")
    table.append(method_name + " & " + " & ".join(values) + r" \\")
    table.append(r"\bottomrule")
    table.append(r"\end{tabular}")
    table.append(r"\end{table}")

    with open(tex_path, 'w') as f:
        f.write("\n".join(table) + "\n")


# ===========================
# 4. One training run
# ===========================

def train_one(args, noise_level=0.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on {device}; terminal noise level = {noise_level:.2%}")

    # Robustness mode should train from scratch for every noise level.
    if args.robustness:
        args.load_ckpt = None

    set_seed(args.seed)
    config = vars(args).copy()
    config['current_noise_level'] = noise_level

    experiment_dir = build_experiment_dir(args, noise_level if args.robustness else None)
    os.makedirs(experiment_dir, exist_ok=True)

    with open(os.path.join(experiment_dir, 'config.yml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    print('-' * 20)
    for key, value in config.items():
        print(f'{key}: {value}')
    print('-' * 20)

    log_file = os.path.join(experiment_dir, 'log.csv')
    best_ckpt_path = os.path.join(experiment_dir, 'model.pth')

    problem = LQR_HJB(args.T, device)

    def sample_interior(sampling_mode):
        if sampling_mode == 'mesh':
            t_mesh = torch.linspace(0, args.T, steps=args.np_i)
            x_mesh = torch.linspace(args.range_min, args.range_max, steps=args.np_i)
            T_grid, X_grid = torch.meshgrid(t_mesh, x_mesh, indexing='ij')
            x_i = torch.stack([T_grid.reshape(-1), X_grid.reshape(-1)]).permute(1, 0)
        else:
            t_rand = torch.rand((args.np_i**2, 1)) * args.T
            x_rand = torch.rand((args.np_i**2, 1)) * (args.range_max - args.range_min) + args.range_min
            x_i = torch.cat([t_rand, x_rand], dim=1)
        return x_i.to(device)

    x_test = sample_interior('mesh')

    # Terminal condition points: t = T.
    # In LQR-HJB, terminal targets are nonzero:
    #   V(T,x) = x^2, u(T,x) = -2x.
    # Therefore, we can directly inject relative Gaussian noise into terminal targets.
    t_T = torch.ones((args.np_b, 1)) * args.T
    x_T_val = torch.rand((args.np_b, 1)) * (args.range_max - args.range_min) + args.range_min
    x_T = torch.cat([t_T, x_T_val], dim=1).to(device)

    with torch.no_grad():
        V_T_clean, u_T_clean = problem.exact_solution(x_T)
        # Use each target's own standard deviation as the perturbation scale.
        # Clamp prevents numerical issues in degenerate settings.
        V_scale = torch.std(V_T_clean).clamp_min(1e-12)
        u_scale = torch.std(u_T_clean).clamp_min(1e-12)

        noise_gen = torch.Generator(device=device)
        noise_gen.manual_seed(args.seed + args.noise_seed_offset + int(round(noise_level * 10000)))

        V_T_target = V_T_clean.clone()
        u_T_target = u_T_clean.clone()
        if noise_level > 0:
            if args.noise_on in ['V', 'both']:
                V_T_target = V_T_target + noise_level * V_scale * torch.randn(V_T_target.shape, generator=noise_gen, device=device)
            if args.noise_on in ['u', 'both']:
                u_T_target = u_T_target + noise_level * u_scale * torch.randn(u_T_target.shape, generator=noise_gen, device=device)

    model = RMultKAN(width=args.width, grid=args.grid, k=args.k, grid_eps=args.grid_eps, seed=args.seed, device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.eta_min)

    best_l2 = float('inf')
    init_step = 0

    if args.load_ckpt and os.path.exists(args.load_ckpt):
        print(f"Loading checkpoint: {args.load_ckpt}")
        ckpt = torch.load(args.load_ckpt, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        init_step = ckpt.get('epoch', 0) + 1
        best_l2 = ckpt.get('l2', float('inf'))
        print(f"Resumed. Best L2: {best_l2:.4e}")

    with open(log_file, 'w', newline='') as f:
        csv.writer(f).writerow([
            'Step', 'PDE_Loss', 'Terminal_Loss', 'L2_Error',
            'LR', 'Grid_Eps', 'Noise_Level', 'Noise_On'
        ])

    mse_loss = nn.MSELoss()
    loss_history = []
    window_size = args.loss_window_size

    print('Start Training...')
    pbar = tqdm(range(init_step, args.steps), desc='Training', ncols=120)

    for step in pbar:
        optimizer.zero_grad()

        x_curr = sample_interior(args.sampling_mode)
        x_in = x_curr.clone().detach().requires_grad_(True)
        output = model(x_in)

        V = output[:, 0:1]
        u_net = output[:, 1:2]

        grads_V = gradients(V, x_in)
        Vt = grads_V[:, 0:1]
        Vx = grads_V[:, 1:2]

        # Control consistency: u = -dV/dx
        loss_control = mse_loss(u_net, -Vx)

        # HJB residual: Vt + 0.5 u^2 + Vx u = 0
        hjb_residual = Vt + 0.5 * (u_net ** 2) + Vx * u_net
        loss_hjb = mse_loss(hjb_residual, torch.zeros_like(hjb_residual))
        pde_loss = loss_hjb + loss_control

        # Terminal condition loss with noisy targets.
        # Clean equations: V(T,x)=x^2 and u(T,x)=-2x.
        # Robustness equations: V(T,x) ~= V_T_target, u(T,x) ~= u_T_target.
        x_T_in = x_T.clone().detach().requires_grad_(True)
        out_T = model(x_T_in)
        V_T_pred = out_T[:, 0:1]
        u_T_pred = out_T[:, 1:2]

        terminal_loss = mse_loss(V_T_pred, V_T_target) + mse_loss(u_T_pred, u_T_target)

        loss = args.pde_weight * pde_loss + args.bc_weight * terminal_loss
        loss_history.append(loss.detach())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # ================= AGU =================
        if args.use_agu and step >= args.agu_start and step <= args.agu_stop and step % args.agu_freq == 0:
            with torch.no_grad():
                err = hjb_residual.squeeze().detach() ** 2 + 1e-9
                prob = err / torch.sum(err)

                N = x_curr.shape[0]
                N_adaptive = int(N * args.agu_ratio)
                idx_adapt = torch.multinomial(prob, N_adaptive, replacement=True)
                x_adapt = x_curr[idx_adapt]

                idx_uniform = torch.randint(0, N, (N - N_adaptive,), device=device)
                x_uniform = x_curr[idx_uniform]

                x_new = torch.cat([x_adapt, x_uniform], dim=0)

                if args.use_agu_eps and len(loss_history) > 2 * window_size:
                    recent_loss = sum(loss_history[-window_size:]) / window_size
                    prev_loss = sum(loss_history[-2 * window_size:-window_size]) / window_size
                    loss_diff = prev_loss - recent_loss

                    if abs(loss_diff) < 1e-3:
                        model.grid_eps = max(0.0, model.grid_eps - 0.005)
                    else:
                        model.grid_eps = min(1.0, model.grid_eps + 0.005)

                model.update_grid(x_new)

            tqdm.write(f"--- Grid Updated. Grid Eps: {model.grid_eps:.3f}, Resample Ratio: {args.agu_ratio:.3f}. ---")

        # --- Evaluation and saving ---
        if step % args.log_freq == 0 or step == args.steps - 1:
            with torch.no_grad():
                out_eval = model(x_test)
                V_pred_eval = out_eval[:, 0:1]
                V_true_eval, _ = problem.exact_solution(x_test)
                l2_error = torch.norm(V_pred_eval - V_true_eval) / torch.norm(V_true_eval)

                current_lr = scheduler.get_last_lr()[0]
                current_eps = model.grid_eps

            pbar.set_description(
                f"PDE: {pde_loss.item():.2e} | TC: {terminal_loss.item():.2e} | "
                f"L2: {l2_error.item():.2e} | Noise: {noise_level:.0%} | Eps: {current_eps:.3f}"
            )

            with open(log_file, 'a', newline='') as f:
                csv.writer(f).writerow([
                    step, pde_loss.item(), terminal_loss.item(), l2_error.item(),
                    current_lr, current_eps, noise_level, args.noise_on
                ])

            if l2_error.item() < best_l2:
                best_l2 = l2_error.item()
                ckpt = {
                    'epoch': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'l2': best_l2,
                    'noise_level': noise_level,
                    'noise_on': args.noise_on,
                }
                torch.save(ckpt, best_ckpt_path)

    print(f"Training Finished. Best clean-test V L2: {best_l2:.4e}.")
    print(f"Results saved to: {experiment_dir}")
    return best_l2, experiment_dir


# ===========================
# 5. Main / robustness driver
# ===========================

def run_robustness(args):
    noise_levels = parse_noise_levels(args.noise_levels)
    base_dir = os.path.join(args.model_dir, args.name)
    os.makedirs(base_dir, exist_ok=True)

    summary_csv = os.path.join(base_dir, 'robustness_summary.csv')
    rows = []

    for nl in noise_levels:
        args_i = copy.deepcopy(args)
        best_l2, exp_dir = train_one(args_i, noise_level=nl)
        rows.append({
            'Noise_Level': nl,
            'Noise_Percent': int(round(nl * 100)),
            'Best_L2_Error': best_l2,
            'Experiment_Dir': exp_dir,
            'Noise_On': args.noise_on,
        })

    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Noise_Level', 'Noise_Percent', 'Best_L2_Error', 'Experiment_Dir', 'Noise_On'])
        writer.writeheader()
        writer.writerows(rows)

    tex_path = os.path.join(base_dir, 'robustness_table.tex')
    write_latex_table(summary_csv, tex_path, method_name='ARMultKAN-PINN' if args.use_agu else 'RMultKAN-PINN')

    print('\nRobustness summary:')
    for row in rows:
        print(f"  noise={row['Noise_Percent']}% | best clean-test V L2={row['Best_L2_Error']:.4e}")
    print(f"Summary CSV saved to: {summary_csv}")
    print(f"LaTeX table saved to: {tex_path}")


def main():
    args = get_args()
    if args.robustness:
        run_robustness(args)
    else:
        train_one(args, noise_level=0.0)


if __name__ == '__main__':
    main()
