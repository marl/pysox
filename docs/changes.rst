Changes
-------

v0.1
~~~~~~

- Initial release.

v1.1.8
~~~~~~
- Move specification of input/output file arguments from __init__ to .build()

v1.3.0
~~~~~~
- patched core sox call to work on Windows
- added remix
- added gain to mcompand
- fixed scientific notation format bug
- allow null output filepaths in `build`
- added ability to capture `build` outputs to stdout and stderr
- added `power_spectrum`
- added `stat`
- added `clear` method
- added `noiseprof` and `noisered` effects
- added `vol` effect
- fixed `Combiner.preview()`