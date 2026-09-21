# SPDX-License-Identifier: Apache-2.0

"""Allow ``python -m turbopairformer`` as well as the installed console script."""

from turbopairformer.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
