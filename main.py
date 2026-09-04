"""
RawForge 入口。

用法:
    python main.py             # 启动图形界面
    python main.py --cli ...   # 命令行模式 (参数见 rawforge.cli)
"""

from __future__ import annotations

import sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        sys.argv.pop(1)
        from rawforge.cli import main as cli_main
        sys.exit(cli_main())
    from rawforge.gui import run
    run()


if __name__ == "__main__":
    main()
