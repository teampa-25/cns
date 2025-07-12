# Avvia un container Docker con supporto GPU e volume condiviso con la repo
# Esegui questo script da PowerShell nella root della repo

$repoPath = (Get-Location).Path

docker run --gpus all -it -v "${repoPath}:/workspace" --workdir /workspace -p 8000:8000 cns-gpu bash -c "apt update && apt install -y dos2unix && dos2unix src/CNS_venv/bin/activate && source src/CNS_venv/bin/activate && bash"
