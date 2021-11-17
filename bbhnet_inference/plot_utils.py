
import logging

import matplotlib as mpl
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logging.getLogger('matplotlib').setLevel(logging.INFO)

def plotPxx(Pxx, freqs, out=None, title=None):
    ''' Function to plot PSD and save '''
    
    fig, ax = plt.subplots(1)
    
    ax.loglog(freqs, Pxx)
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('PSD [strain / Hz]')
    ax.set_xlim(10, None)
    if title is not None:
        ax.set_title(title)
    
    fig.tight_layout()
    
    if out is not None:
        fig.savefig(out, bbox_inches='tight', dpi=300)
    else:
        plt.show()
    plt.close()
