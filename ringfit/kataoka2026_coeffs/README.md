# Kitade & Kataoka 2026 Stokes-I coefficients

`kataoka2026_stokesI_coeffs.npz` is generated from the public
`emergentintensity` repository:

- source: https://github.com/naoyakitade-astro/emergentintensity
- source commit: `75fca711a04ecec5ba122bcc18ced803f6646f8e`
- source data: `data/StokesI_emergent/*.inp`
- fitting form: `src/stokes_i_fitting.py::stokes_i_model_with_mu`

The table stores `A_I`, `B_I`, `I_conv`, and `omega_I` on the official
`omega = 0.0 ... 0.9` grid and an inclination grid from 0 to 80 degrees
converted to `mu = cos(i)`. It is loaded locally at runtime; `dring` does not
download data or refit coefficients during model evaluation.

This bundled table is included only as a convenience/reference implementation
for comparison with the `zhu2019` paper-release default. It should not be
treated as an authoritative replacement for the Kitade & Kataoka release. For
scientific fits that depend on the Kataoka2026 formula, use the official
`emergentintensity` data/code and validate or regenerate the coefficients for
the specific analysis.
