#!/bin/bash
#
# 2I BOUNDARY: ALL SUB-SECTORS (sequential, no memory collision)
# ==============================================================
# Kill the hung 9-irrep run first:
#   ps aux | grep icosa_2I | grep -v grep
#   kill <PID>
#
# Then:
#   chmod +x run_all_2I_sectors.sh
#   nohup ./run_all_2I_sectors.sh > 2I_all_sectors.log 2>&1 &
#
# Output files (all unique):
#   icosa_2I_bosonic_progress.json       (125×125,  ~15 min)
#   icosa_2I_fermionic_progress.json     (64×64,    ~5 min)
#   icosa_2I_lepton_photon_progress.json (216×216,  ~2 hours)
#   icosa_2I_dark_fermi_progress.json    (125×125,  ~15 min)
#
# Estimated total: ~3 hours at 60 cores
# ==============================================================

CORES=30
SCRIPT=icosa_boundary_2I.py

echo "============================================================"
echo "2I BOUNDARY: ALL SUB-SECTORS"  
echo "Start: $(date)"
echo "Cores: $CORES, RAM: $(free -g | awk '/Mem:/{print $2}')G"
echo "============================================================"

# --- SECTOR 1: Bosonic ---
echo ""
echo "=== 1/4 BOSONIC (lifts of A5, 125x125) ==="
echo "$(date): starting..."
python3.14 $SCRIPT --cores $CORES --bosonic
echo "$(date): done"

# --- SECTOR 2: Fermionic ---  
echo ""
echo "=== 2/4 FERMIONIC (spinors, 64x64) ==="
echo "$(date): starting..."
python3.14 $SCRIPT --cores $CORES --fermionic
echo "$(date): done"

# --- SECTOR 3: Lepton+Photon ---
echo ""
echo "=== 3/4 LEPTON+PHOTON (216x216) ==="
echo "$(date): starting..."
rm -f icosa_2I_custom_progress.json icosa_2I_custom_results.json
python3.14 $SCRIPT --cores $CORES --sector 0,1,2,3,4,7
mv icosa_2I_custom_progress.json icosa_2I_lepton_photon_progress.json 2>/dev/null
mv icosa_2I_custom_results.json icosa_2I_lepton_photon_results.json 2>/dev/null
echo "$(date): done"

# --- SECTOR 4: Dark Fermionic ---
echo ""
echo "=== 4/4 DARK FERMIONIC (125x125) ==="
echo "$(date): starting..."
rm -f icosa_2I_custom_progress.json icosa_2I_custom_results.json
python3.14 $SCRIPT --cores $CORES --sector 1,2,6,7,8
mv icosa_2I_custom_progress.json icosa_2I_dark_fermi_progress.json 2>/dev/null
mv icosa_2I_custom_results.json icosa_2I_dark_fermi_results.json 2>/dev/null
echo "$(date): done"

echo ""
echo "============================================================"
echo "ALL COMPLETE: $(date)"
echo "============================================================"
ls -lh icosa_2I_*_progress.json icosa_2I_*_results.json 2>/dev/null
