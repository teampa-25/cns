#!/bin/bash
# Script per creare un virtual environment Python in CNS-devel e installare le dipendenze

# Vai nella cartella dello script
cd "$(dirname "$0")"


# Crea il virtual environment nella cartella CNS_venv
python3 -m venv CNS_venv

# Attiva il virtual environment
source CNS_venv/bin/activate

# Aggiorna pip
pip install --upgrade pip

# Installa i requirements
pip install -r requirements.txt

echo "Virtual environment creato e requirements installati."
