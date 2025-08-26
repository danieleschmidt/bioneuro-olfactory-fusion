"""Kenyon cell models for sparse coding in the mushroom body."""

from .kenyon_cells import (
    KenyonCellLayer,
    AdaptiveKenyonLayer,
    CompetitiveKenyonNetwork,
    MushroomBodyKenyonCells,
    KenyonConfig,
    create_kenyon_cell_system,
    create_competitive_kenyon_network,
    SparseKenyonCells,
    CompetitiveKenyonCells,
    MushroomBodyKCs
)

# Legacy compatibility
AdaptiveKenyonCells = AdaptiveKenyonLayer
KenyonCellConfig = KenyonConfig
HierarchicalKenyonCells = CompetitiveKenyonNetwork
create_sparse_kenyon_network = create_kenyon_cell_system
create_efficient_kenyon_network = create_competitive_kenyon_network

__all__ = [
    'KenyonCellLayer',
    'AdaptiveKenyonLayer',
    'CompetitiveKenyonNetwork', 
    'MushroomBodyKenyonCells',
    'KenyonConfig',
    'create_kenyon_cell_system',
    'create_competitive_kenyon_network',
    'SparseKenyonCells',
    'CompetitiveKenyonCells',
    'MushroomBodyKCs',
    # Legacy exports
    'AdaptiveKenyonCells',
    'KenyonCellConfig',
    'HierarchicalKenyonCells',
    'create_sparse_kenyon_network',
    'create_efficient_kenyon_network'
]