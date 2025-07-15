# Avvia un container Docker con supporto GPU e volume condiviso con la repo
# Esegui questo script da PowerShell nella root della repo

$repoPath = (Get-Location).Path


# old docker versions
# docker run --gpus all -it --runtime=nvidia -v "${repoPath}:/workspace" --workdir /workspace -p 8000:8000 humongous-cns bash -c "apt update && apt install -y dos2unix && dos2unix src/CNS_venv/bin/activate && source src/CNS_venv/bin/activate && bash"

# sotto il comando precedente
docker run --gpus all -it -v "${repoPath}:/workspace" --name cns-powa --workdir /workspace -p 8000:8000 cns-powa bash -c "apt update && apt install -y dos2unix && dos2unix src/CNS_venv/bin/activate && source src/CNS_venv/bin/activate && bash"
