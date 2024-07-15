import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import mcubes
import trimesh


# Positional encoding embedding. Code was borrowed from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': multires-1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


class BackgroundModel(nn.Module):
    def __init__(self):
        super().__init__()
        n_neurons = 128
        n_hidden_layers = 4
        self.embedder, dims = get_embedder(2)
        modules = [
            nn.Linear(dims, n_neurons),
            nn.Softplus(beta=20)
        ]
        for _ in range(n_hidden_layers - 1):
            modules.append(
                nn.Linear(n_neurons, n_neurons))
            modules.append(nn.Softplus(beta=20))
        modules.append(nn.Linear(n_neurons, 4))
        self.model = nn.Sequential(*modules)

    def forward(self, points):
        embedded = self.embedder(points)
        out_sdf, out_rgb = torch.split(self.model(embedded), [1, 3], dim=-1)
        out_rgb = torch.sigmoid(out_rgb)
        return out_sdf, out_rgb

    def fit(self, points, colors, device, batch_size=50000, niter=1500):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
        pts_zero = torch.tensor(points, device=device, dtype=torch.float32)
        rgb_zero = torch.tensor(colors / 255, device=device, dtype=torch.float32)
        pbar = tqdm(range(niter))
        for i in pbar:
            batch_idx = torch.randint(0, pts_zero.shape[0], (batch_size,))
            optimizer.zero_grad()
            direction = torch.randn_like(pts_zero[batch_idx][:, 0]) * 0.1 + 1
            pts_rand = pts_zero[batch_idx] * direction[:, None]
            sdf = torch.linalg.norm(pts_rand - pts_zero[batch_idx], dim=1, keepdim=True)
            sdf[direction < 1] *= -1
            out_sdf, out_rgb = self.forward(pts_rand)
            loss_sdf = F.mse_loss(out_sdf, sdf)
            loss_rgb = F.mse_loss(rgb_zero[batch_idx], out_rgb)
            loss = loss_sdf + loss_rgb * 5
            loss.backward()
            optimizer.step()
            scheduler.step()
            pbar.set_description(f"Background reconstruction loss: {loss.item():.4f} ")

    @torch.no_grad()
    def extract_mesh(self, lower_bounds, upper_bounds, device, frustum_planes=None, resolution=256, margin=0.05):
        lower_bounds -= margin
        upper_bounds += margin
        grid_range = upper_bounds - lower_bounds
        grid = np.stack(np.meshgrid(*(
                np.linspace(0, 1, resolution)[None] * grid_range[:, None] + lower_bounds[:, None]
        ), indexing='ij'), axis=-1)
        out_sdf = np.concatenate([
            self.forward(batch)[0].squeeze(-1).cpu().numpy()
            for batch in torch.tensor(grid, device=device, dtype=torch.float32).view(-1, 3).split(100000)
        ])
        vertices, triangles = mcubes.marching_cubes(np.float32(out_sdf.reshape(grid.shape[:-1])), 0)
        vertices /= (resolution - 1)
        vertices *= grid_range
        vertices += lower_bounds
        mesh = trimesh.Trimesh(vertices, triangles)
        for point_normal in frustum_planes:
            mesh = mesh.slice_plane(point_normal[:3], point_normal[3:])
        colors = self.forward(torch.tensor(mesh.vertices, device=device, dtype=torch.float32))[1].cpu().numpy()
        mesh.visual.vertex_colors = np.uint8(colors * 255)
        return mesh





