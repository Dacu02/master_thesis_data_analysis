import numpy as np

def SNR(x_ref: np.ndarray, x_est: np.ndarray):
    """
    Compute the reconstruction signal-to-noise ratio (SNR) between the reference signal and the estimated signal.
    
    Parameters
    ----------
    x_ref : np.ndarray
        The reference signal.
    x_est : np.ndarray
        The estimated signal.
    remove_mean : bool, optional
        If True, remove the mean from both signals before computing the SNR. Default is True
    """
    x_ref = np.asarray(x_ref, dtype=float)
    x_est = np.asarray(x_est, dtype=float)

    if x_ref.shape != x_est.shape:
        raise ValueError("I due segnali devono avere la stessa forma/lunghezza.")

    err = x_ref - x_est

    signal_power = np.mean(x_ref ** 2)
    noise_power = np.mean(err ** 2)

    return 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')


def MSE(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute the mean squared error (MSE) between the signal and noise.

    Parameters
    ----------
    signal : np.ndarray
        The signal array.
    noise : np.ndarray
        The noise array.

    Returns
    -------
    mse : float
        The mean squared error.
    """
    if len(signal) != len(noise):
        raise ValueError("Signal and noise arrays must have the same length.")
    mse = float(np.mean((signal - noise) ** 2))
    return mse

def HL(signal: np.ndarray, reconstruction: np.ndarray, N:int, tilde_N:int) -> float:
    """
    Rosanna Coccaro, Enrico Ferrentino, Antonio Parziale, Angelo Marcelli, Pasquale Chiacchio
    
    Compute the Human Likeness Index (HL) between the signal and the reconstructed signal.

    Parameters
    ----------
    signal : np.ndarray
        The recorded signal array.
    reconstruction : np.ndarray
        The reconstructed signal array.
    N : int
        The number of strokes in the signal.
    tilde_N : int
        The expected number of strokes in the signal

    Returns
    -------
    hl : float
        The Human Likeness Index.
    """
    if tilde_N < N:
        return float('inf')
    return SNR(signal, reconstruction) * tilde_N / N
