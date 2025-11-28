from typing import Optional

__all__ = ['__version__', 'debug', 'cuda', 'git_version', 'hip']
__version__ = '2.6.0+cu124'
debug = False
cuda: Optional[str] = '12.4'
git_version = '2236df1770800ffea5697b11b0bb0d910b2e59e1'
hip: Optional[str] = None
xpu: Optional[str] = None
