import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.01)

    def forward(self, logits, labels):
        return self.cross_entropy(logits, labels)


class HierarchicalProxyLoss(nn.Module):
    """
    Hierarchical Proxy Loss for DCASE Task 1
    Combines 5 loss terms for hierarchical audio classification
    
    Args:
        embedding_dim: Dimension of embeddings (default: 64)
        num_parents: Number of top-level classes (default: 5)
        num_children: Number of subclasses (default: 23)
        temperature: Temperature for softmax sharpening (default: 0.07)
        alpha: Weight for parent classification loss (default: 0.4)
        beta: Weight for child-to-parent alignment loss (default: 0.3)
        gamma: Weight for sibling separation loss (default: 0.15)
        delta: Weight for parent separation loss (default: 0.05)
        sibling_margin: Margin for sibling separation (default: 0.4)
        parent_margin: Margin for parent separation (default: 0.0)
    """
    
    def __init__(
        self,
        embedding_dim=64,
        num_parents=5,
        num_children=23,
        temperature=0.07,
        alpha=0.4,
        beta=0.3,
        gamma=0.15,
        delta=0.05,
        sibling_margin=0.4,
        parent_margin=0.0,
    ):
        super().__init__()
        
        self.num_parents = num_parents
        self.num_children = num_children
        self.temperature = temperature
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        
        self.sibling_margin = sibling_margin
        self.parent_margin = parent_margin
        
        # Initialize proxy vectors (learnable parameters)
        self.parent_proxies = nn.Parameter(torch.randn(num_parents, embedding_dim))
        self.child_proxies = nn.Parameter(torch.randn(num_children, embedding_dim))
        
        # Initialize with Xavier uniform
        nn.init.xavier_uniform_(self.parent_proxies)
        nn.init.xavier_uniform_(self.child_proxies)
    
    def forward(self, z, parent_labels, child_labels, parent_of_child):
        """
        Args:
            z: [B, D] - Normalized embeddings
            parent_labels: [B] - Parent class indices
            child_labels: [B] - Child class indices
            parent_of_child: [num_children] - Mapping from child to parent
        
        Returns:
            total_loss: Scalar loss
            loss_dict: Dictionary with individual loss components
        """
        
        # Normalize proxies
        parent_proxies = F.normalize(self.parent_proxies, dim=1)
        child_proxies = F.normalize(self.child_proxies, dim=1)
        
        # ===================================================
        # Loss 1: Child classification loss
        # ===================================================
        child_logits = torch.matmul(z, child_proxies.T) / self.temperature
        loss_child_cls = F.cross_entropy(child_logits, child_labels)
        
        # ===================================================
        # Loss 2: Parent classification loss
        # ===================================================
        parent_logits = torch.matmul(z, parent_proxies.T) / self.temperature
        loss_parent_cls = F.cross_entropy(parent_logits, parent_labels)
        
        # ===================================================
        # Loss 3: Child proxy -> corresponding parent proxy alignment
        # ===================================================
        child_parent_proxy = parent_proxies[parent_of_child]  # [num_children, D]
        child_to_parent_sim = torch.sum(child_proxies * child_parent_proxy, dim=1)
        loss_child_to_parent = torch.mean(1.0 - child_to_parent_sim)
        
        # ===================================================
        # Loss 4: Same-parent child proxy separation
        # ===================================================
        child_sim = torch.matmul(child_proxies, child_proxies.T)  # [num_children, num_children]
        
        same_parent = parent_of_child.unsqueeze(0) == parent_of_child.unsqueeze(1)
        not_self = ~torch.eye(
            self.num_children,
            dtype=torch.bool,
            device=z.device
        )
        
        sibling_mask = same_parent & not_self
        sibling_sim = child_sim[sibling_mask]
        
        if sibling_sim.numel() > 0:
            loss_sibling_sep = F.relu(sibling_sim - self.sibling_margin).mean()
        else:
            loss_sibling_sep = torch.tensor(0.0, device=z.device)
        
        # ===================================================
        # Loss 5: Parent proxy separation
        # ===================================================
        parent_sim = torch.matmul(parent_proxies, parent_proxies.T)
        
        parent_not_self = ~torch.eye(
            self.num_parents,
            dtype=torch.bool,
            device=z.device
        )
        
        parent_pair_sim = parent_sim[parent_not_self]
        
        if parent_pair_sim.numel() > 0:
            loss_parent_sep = F.relu(parent_pair_sim - self.parent_margin).mean()
        else:
            loss_parent_sep = torch.tensor(0.0, device=z.device)
        
        # ===================================================
        # Total loss
        # ===================================================
        total_loss = (
            loss_child_cls
            + self.alpha * loss_parent_cls
            + self.beta * loss_child_to_parent
            + self.gamma * loss_sibling_sep
            + self.delta * loss_parent_sep
        )
        
        loss_dict = {
            "total": total_loss.item(),
            "child_cls": loss_child_cls.item(),
            "parent_cls": loss_parent_cls.item(),
            "child_to_parent": loss_child_to_parent.item(),
            "sibling_sep": loss_sibling_sep.item() if isinstance(loss_sibling_sep, torch.Tensor) else loss_sibling_sep,
            "parent_sep": loss_parent_sep.item() if isinstance(loss_parent_sep, torch.Tensor) else loss_parent_sep,
        }
        
        return total_loss, loss_dict