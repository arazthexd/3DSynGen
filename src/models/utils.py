import torch

def get_edge_directions(pos: torch.Tensor, 
                        edge_index: torch.Tensor,
                        normalize: bool = True, 
                        eps: float = 1e-8) -> torch.Tensor:
    
    directions = pos[edge_index[1]] - pos[edge_index[0]]
        
    if normalize:
        norms = torch.norm(directions, dim=1, keepdim=True)
        directions = directions / (norms + eps) # avoid zero division
    
    return directions.reshape(-1, 3)

def get_edge_rbf(pos: torch.Tensor, 
                 edge_index: torch.Tensor, 
                 D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    (Modified) From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''

    D = torch.norm(pos[edge_index[1]]-pos[edge_index[0]], dim=1)
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF