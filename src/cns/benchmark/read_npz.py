import numpy as np
import os
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#folder_path = "results/compare_with_image_vs_on_mscoco14_150/06_21_13_54_06_CNS/scale=1.00"
folder_path = "../../"
def read_npz_files(folder_path):
    npz_files = [f for f in os.listdir(folder_path) if f.endswith('.npz')]
    data_list = []
    file_names = []
    for file in npz_files:
        file_path = os.path.join(folder_path, file)
        data = np.load(file_path)
        data_list.append(data)
        file_names.append(file)
    return data_list, file_names

def analyze_sim_Erender_result(npz_path="sim_Erender_result.npz"):
    data = np.load(npz_path)
    print("Chiavi disponibili:", data.files)
    print("Shape initial_pose:", data['initial_pose'].shape)
    print("Shape final_pose:", data['final_pose'].shape)
    print("Shape desired_pose:", data['desired_pose'].shape)
    print("Shape steps:", data['steps'].shape)
    print("Shape gid:", data['gid'].shape)

    # Errore di traslazione finale rispetto al target
    trans_errors = np.linalg.norm(data['final_pose'][:, :3, 3] - data['desired_pose'][:, :3, 3], axis=1)
    print("Errore di traslazione medio:", np.mean(trans_errors))
    print("Errore di traslazione per round:", trans_errors)

    # Steps medi
    print("Steps medi:", np.mean(data['steps']))
    print("Steps per round:", data['steps'])

    # Plot errore di traslazione
    plt.figure()
    plt.plot(trans_errors, label='Errore traslazione')
    plt.xlabel('Round')
    plt.ylabel('Errore traslazione (m)')
    plt.title('Errore finale per round')
    plt.legend()
    plt.tight_layout()
    plt.savefig("sim_Erender_error.png")
    print("Plot salvato come sim_Erender_error.png")

def analyze_folder(folder_path):
    data_list, file_names = read_npz_files(folder_path)
    final_errors = []
    traj_lengths = []
    all_errors = []

    for i, data in enumerate(data_list):
        print(f"Data from file {file_names[i]}:")
        print(data.files)  # Print the keys in the npz file
        if 'errors' in data:
            errors = data['errors']
            final_errors.append(errors[-1])
            all_errors.append(errors)
            print(f"Errore finale: {errors[-1]}")
            print(f"Lunghezza traiettoria: {len(errors)})")
        if 'trajs' in data:
            print(f"Shape traiettoria: {data['trajs'].shape}")
        print("\n")

    if final_errors:
        print(f"Errore finale medio su {len(final_errors)} file: {np.mean(final_errors)}")
    else:
        print("Nessun errore trovato nei file npz.")

    # Plot degli errori per ogni file e salva su PNG
    if all_errors:
        for i, errors in enumerate(all_errors):
            plt.plot(errors, label=file_names[i])
        plt.xlabel("Step")
        plt.ylabel("Errore")
        plt.title("Andamento errore per file")
        plt.legend()
        plt.tight_layout()
        plt.savefig("andamento_errori.png")
        print("Grafico salvato come andamento_errori.png")

if __name__ == "__main__":
    # Uso: python read_npz.py [sim|folder] [path]
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "sim":
            npz_path = sys.argv[2] if len(sys.argv) > 2 else "sim_Erender_result.npz"
            print(f"Analisi sim_Erender_result: {npz_path}")
            analyze_sim_Erender_result(npz_path)
        elif mode == "folder":
            folder = sys.argv[2] if len(sys.argv) > 2 else folder_path
            print(f"Analisi folder: {folder}")
            analyze_folder(folder)
        else:
            print("Argomento non riconosciuto. Usa 'sim' o 'folder'.")
    else:
        # Default: analizza la cartella
        analyze_folder(folder_path)
        # E se esiste, anche il file sim_Erender_result.npz
        if os.path.exists("../../sim_Erender_result.npz"):
            print("\nAnalisi sim_Erender_result.npz:")
            analyze_sim_Erender_result("../../sim_Erender_result.npz")