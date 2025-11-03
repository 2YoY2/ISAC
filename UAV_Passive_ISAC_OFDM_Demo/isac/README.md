
# UAV Passive ISAC — Opportunistic Radar on OFDM (Minimal Demo)

This repo contains a small, runnable demonstration:
- Generate an OFDM waveform.
- Create a surveillance channel with delay/Doppler targets and clutter.
- Compute a range–Doppler (RD) map using blockwise cross-correlation and slow-time FFT.
- Plot the RD map.

## Run
```bash
python -m sims.uav_passive
```

## Structure
- `waveform/ofdm.py` — OFDM baseband generator with comb pilots.
- `channel/delay_doppler.py` — fractional delay + Doppler, surveillance mixer.
- `radar_rx/rdmap.py` — block cross-correlation + Doppler FFT → RD map.
- `radar_rx/cfar.py` — simple 2D CA-CFAR (optional).
- `metrics/plots.py` — basic matplotlib plotting helpers.
- `sims/uav_passive.py` — end-to-end demo script.
