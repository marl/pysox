Changes
-------

v1.4.0
~~~~~~
- added `.build_array()` which supports file or in memory inputs and array outputs
- added `.build_file()` - an alias to `.build()`
- refactored `.build()` function to support file or in-memory array inputs and file outputs
- the call to subprocess calls the binary directly (shell=False)
- file_info methods return None instead of 0 when the value is not available
- fixed bug in `file_info.bitrate()`, which was returning bitdepth
- added `file_info.bitdepth()`
- added Windows support for `soxi`
- added configurable logging
- `.trim()` can be called with only the start time specificed

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

v1.1.8
~~~~~~
- Move specification of input/output file arguments from __init__ to .build()

v0.1
~~~~~~

- Initial release.