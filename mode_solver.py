import numpy as np
import matplotlib.pyplot as plt

def setup_environment():
    """
    Define sound speed and density profiles
    c_w, rho_w = 1480, 1 # m/s, g/cm^3
    c_b, rho_b = 1650, 1.8
    set frequency to be 50 Hz
    Water column = 500 m
    Bottom layer = 100 m
    in the bottom layer: attenuation = 0.2
    water there is no attenuation
    define Boundary Conditions
    Top - z = 0 : Pressure release (i.e., pressure = 0)
    Interface - z = 500 : Continuity of pressure and normal velocity
    Bottom - z = 600 : Radiation condition (allowing energy to leak out, not reflect)
    divide the ocean into layers for finite difference method
    """
    # Set parameters
    freq = 50.0  # Hz
    omega = 2 * np.pi * freq

    # Depth grid
    dz = 1.0  # depth resolution (m)
    z_max = 600  # maximum depth (m)
    z = np.arange(0, z_max + dz, dz)
    n_points = len(z)

    # Sound speed profile (isovelocity in each layer)
    c = np.ones(n_points) * 1480.0  # water sound speed
    c[z > 500] = 1650.0  # bottom sound speed

    # Density profile
    rho = np.ones(n_points)*1000  # water density
    rho[z > 500] = 1800  # bottom density

    # Attenuation profile (converted to Np/m from dB/λ)
    alpha = np.zeros(n_points)
    bottom_indices = z > 500
    wavelength_b = 1650.0 / freq  # wavelength in bottom (m)
    # Convert from dB/λ to Np/m: 0.2 dB/λ * (8.686 Np/dB)^-1 * (1/λ)
    alpha[bottom_indices] = 0.2 / 8.686 / wavelength_b

    # Complex sound speed due to attenuation
    k = omega / c * (1 + 1j * alpha * c / omega)

    return z, c, rho, k, omega, freq

def compute_mode_shapes():
    """
    Setup finite difference method to compute mode shapes and eigenvalues
    Using regular NumPy arrays instead of SciPy sparse matrices
    """
    z, c, rho, k, omega, freq = setup_environment()
    dz = z[1] - z[0]
    n_points = len(z)

    # Create finite difference matrix for the wave equation
    A = np.zeros((n_points, n_points), dtype=complex)

    # Set up second derivative operator with central differences
    for i in range(1, n_points-1):
        A[i, i-1] = 1.0 / dz**2
        A[i, i+1] = 1.0 / dz**2
        A[i, i] = -2.0 / dz**2

    # Apply boundary conditions
    # At z=0: Pressure release (p=0) - Dirichlet condition
    A[0, 0] = 1.0
    A[0, 1:] = 0.0

    # At z=z_max: Radiation condition (approximate with soft bottom)
    A[n_points-1, n_points-1] = 1.0 - 1j * np.real(k[-1]) * dz
    A[n_points-1, n_points-2] = -1.0

    # Create the k^2(z) diagonal matrix
    K = np.diag(k**2)

    # The normal mode equation: d^2Φ/dz^2 + [k^2(z) - kr^2]Φ = 0
    # For eigenvalue problem: AΦ = λΦ where λ = -kr^2
    # A = -d^2/dz^2 - k^2(z)
    H = -A - K  # Negative sign because eigenvalues = -kr^2

    # Solve eigenvalue problem
    eigenvalues, eigenvectors = np.linalg.eig(H)

    # Convert eigenvalues to horizontal wavenumbers squared
    kr_squared = -eigenvalues

    # Find propagating modes (real part of kr_squared > 0)
    propagating_indices = np.where(np.real(kr_squared) > 0)[0]
    eigenvalues_prop = eigenvalues[propagating_indices]
    eigenvectors_prop = eigenvectors[:, propagating_indices]
    kr_squared_prop = kr_squared[propagating_indices]
    kr_prop = np.sqrt(kr_squared_prop + 0j)

    # Sort by increasing real part of eigenvalue (related to decreasing phase speed/increasing mode number)
    sort_idx = np.argsort(np.real(eigenvalues_prop))
    eigenvalues_sorted = eigenvalues_prop[sort_idx]
    eigenvectors_sorted = eigenvectors_prop[:, sort_idx]
    kr_sorted = kr_prop[sort_idx]

    # Select the first 10 modes based on this initial sorting
    num_modes_to_analyze = min(10, len(eigenvalues_sorted))
    kr_analyze = kr_sorted[:num_modes_to_analyze]
    mode_shapes_analyze = eigenvectors_sorted[:, :num_modes_to_analyze]

    # Count nodes (zero crossings) within the water column (excluding boundaries)
    node_counts = []
    water_column_indices = np.where((z > 10) & (z < 490))[0] # Exclude near boundaries
    for i in range(mode_shapes_analyze.shape[1]):
        mode_real = np.real(mode_shapes_analyze[water_column_indices, i])
        crossings = np.sum(np.diff(np.signbit(mode_real)) != 0)
        node_counts.append(crossings)

    # Sort modes based on the number of nodes (ascending)
    node_sort_idx = np.argsort(node_counts)
    kr = kr_analyze[node_sort_idx]
    mode_shapes = mode_shapes_analyze[:, node_sort_idx]

    # Take the first 6 modes
    num_modes = min(6, mode_shapes.shape[1])
    kr = kr[:num_modes]
    mode_shapes = mode_shapes[:, :num_modes]

    # Normalize mode shapes 
    for j in range(mode_shapes.shape[1]):
        mode_real = np.real(mode_shapes[:, j])
        norm_factor = np.max(np.abs(mode_real))
        if norm_factor > 1e-9:
            mode_shapes[:, j] = mode_shapes[:, j] / norm_factor

    return z, c, rho, kr, mode_shapes, freq


def plot_mode_shapes():
    """ Plot the normalized mode shape solutions for 6 modes """
    z, c, rho, kr, mode_shapes, freq = compute_mode_shapes()

    # Get real part of mode shapes for plotting
    mode_shapes_real = np.real(mode_shapes)

    # Number of modes to plot
    num_modes = mode_shapes.shape[1]

    # Create a figure with subplots - one for each mode plus sound speed profile
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1])

    # Plot sound speed profile
    ax_ssp = fig.add_subplot(gs[:, 0])
    ax_ssp.plot(c.real, z, 'b-', linewidth=2)
    ax_ssp.set_xlabel('Sound Speed (m/s)')
    ax_ssp.set_ylabel('Depth (m)')
    ax_ssp.set_title('Sound Speed Profile')
    ax_ssp.grid(True)
    ax_ssp.invert_yaxis()  # Oceanographic convention: depth increases downward
    ax_ssp.axhline(y=500, color='k', linestyle='--', label='Water-Bottom Interface')
    ax_ssp.axhline(y=600, color='r', linestyle='--', label='Bottom Boundary')
    ax_ssp.legend()

    # Create axes for modes in the correct order
    axes = []
    for i in range(num_modes):
        row = i // 3
        col = (i % 3) + 1
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

    # Plot each mode
    for i in range(num_modes):
        mode = mode_shapes_real[:, i]

        # Plot mode shape
        axes[i].plot(mode, z, 'b-', linewidth=2)

        # Fill areas for positive/negative phases
        axes[i].fill_betweenx(z, 0, mode, where=(mode > 0), color='blue', alpha=0.3)
        axes[i].fill_betweenx(z, mode, 0, where=(mode < 0), color='red', alpha=0.3)

        # Add reference lines
        axes[i].axvline(x=0, color='k', linestyle='-', alpha=0.3)
        axes[i].axhline(y=500, color='k', linestyle='--', linewidth=1)  # Water-bottom interface

        # Set axis limits
        axes[i].set_xlim(-1.1, 1.1)
        axes[i].set_ylim(0, 600)
        axes[i].invert_yaxis()

        # Add grid, labels and title
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].set_xlabel('Normalized Amplitude')
        axes[i].set_title(f'Mode {i+1}')

        # Only add y-label for leftmost plots
        if i % 3 == 0:
            axes[i].set_ylabel('Depth (m)')

    # Add a common legend
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], color='k', linestyle='--', label='Water-Bottom Interface (500m)'),
        mpatches.Patch(color='blue', alpha=0.3, label='Positive Phase'),
        mpatches.Patch(color='red', alpha=0.3, label='Negative Phase')
    ]

    fig.legend(handles=legend_items, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.02))

    plt.suptitle(f"Normal Modes (f = {freq} Hz)", fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0.12, 1, 0.95])

    return fig

# Execute the analysis and display the results
if __name__ == "__main__":
    fig_combined = plot_mode_shapes()
    plt.show()