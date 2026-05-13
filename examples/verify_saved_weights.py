import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import portpy.photon as pp

from echo_vmat.arcs import Arcs
from echo_vmat.utils.get_sparse_only import get_sparse_only


# Default settings. Edit these paths if your outputs are stored elsewhere.
DATA_DIR = os.path.join(".", "data", "data")
PATIENT_ID = "Lung_Phantom_Patient_1"
PROTOCOL_NAME = "Lung_2Gy_30Fx"
SOLUTION_DIR = os.path.join( "Temp", PATIENT_ID)
PLOT_DVH = True


def build_plan(
    data_dir: str,
    patient_id: str,
    protocol_name: str,
):
    data = pp.DataExplorer(data_dir=data_dir)
    data.patient_id = patient_id

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config_dir = os.path.join(repo_root, "echo_vmat", "config_files")

    opt_params = data.load_json(
        file_name=os.path.join(config_dir, f"{protocol_name}_opt_params.json")
    )
    clinical_criteria = pp.ClinicalCriteria(
        file_name=os.path.join(config_dir, f"{protocol_name}_clinical_criteria.json")
    )

    flag_full_matrix = opt_params["opt_parameters"].get("flag_full_matrix", False)

    structs = pp.Structures(data)
    if "Patient Surface" in structs.get_structures():
        ind = structs.structures_dict["name"].index("Patient Surface")
        structs.structures_dict["name"][ind] = "BODY"

    for i in range(len(structs.structures_dict["name"])):
        structs.structures_dict["name"][i] = structs.structures_dict["name"][i].upper()

    for i in range(len(opt_params["steps"])):
        structs.create_opt_structures(
            opt_params=opt_params["steps"][str(i + 1)],
            clinical_criteria=clinical_criteria,
        )

    all_beam_ids = np.arange(0, 37)
    arcs_dict = {
        "arcs": [
            {"arc_id": "01", "beam_ids": all_beam_ids[0 : int(len(all_beam_ids) / 2)]},
            {"arc_id": "02", "beam_ids": all_beam_ids[int(len(all_beam_ids) / 2) :]},
        ]
    }

    beam_ids = [beam_id for arc in arcs_dict["arcs"] for beam_id in arc["beam_ids"]]
    beams = pp.Beams(data, beam_ids=beam_ids, load_inf_matrix_full=flag_full_matrix)
    inf_matrix = pp.InfluenceMatrix(structs=structs, beams=beams, is_full=flag_full_matrix)

    if flag_full_matrix:
        full_matrix = deepcopy(inf_matrix.A)
        threshold_perc = opt_params["opt_parameters"].get("threshold_perc", 5)
        sparsification = opt_params["opt_parameters"].get("sparsification", "Naive")
        inf_matrix.A = get_sparse_only(
            full_matrix, threshold_perc=threshold_perc, compression=sparsification
        )

    scale_factor = opt_params["opt_parameters"].get("inf_matrix_scale_factor", 1)
    inf_matrix.A = inf_matrix.A * np.float32(scale_factor)

    arcs = Arcs(arcs_dict=arcs_dict, inf_matrix=inf_matrix)
    if "voxel_coordinate_XYZ_mm" not in inf_matrix.opt_voxels_dict:
        inf_matrix.opt_voxels_dict["voxel_coordinate_XYZ_mm"] = [inf_matrix.get_voxel_coordinates()]

    my_plan = pp.Plan(
        structs=structs,
        beams=beams,
        inf_matrix=inf_matrix,
        clinical_criteria=clinical_criteria,
        arcs=arcs,
    )
    return my_plan, inf_matrix


def load_solution(sol_path: str):
    sol_dir = os.path.dirname(sol_path) or "."
    sol_name = os.path.basename(sol_path)
    sol = pp.load_optimal_sol(sol_name=sol_name, path=sol_dir)
    if "optimal_intensity" not in sol:
        raise KeyError(f"'optimal_intensity' not found in solution file: {sol_path}")
    return sol


def compute_dose(my_plan, inf_matrix, weights):
    if len(weights) != inf_matrix.A.shape[1]:
        raise ValueError(
            f"Weight length mismatch: got {len(weights)}, expected {inf_matrix.A.shape[1]}. "
            "Check patient, beams/arcs, and influence matrix settings."
        )
    return inf_matrix.A @ weights * my_plan.get_num_of_fractions()


def get_solution_files(sol_dir):
    candidates = []
    preferred_names = [
        "sol_col_gen",
        "sol_col_gen.pkl",
        "sol_step0.pkl",
        "sol_step1.pkl",
        "sol_step2.pkl",
    ]
    for name in preferred_names:
        path = os.path.join(sol_dir, name)
        if os.path.exists(path):
            candidates.append(path)

    repo_temp_dir = os.path.join(os.path.dirname(__file__), "Temp")
    for name in ["sol_col_gen", "sol_col_gen.pkl"]:
        path = os.path.join(repo_temp_dir, name)
        if os.path.exists(path):
            abs_path = os.path.normcase(os.path.abspath(path))
            if abs_path not in {
                os.path.normcase(os.path.abspath(existing_path)) for existing_path in candidates
            }:
                candidates.append(path)

    if os.path.isdir(sol_dir):
        existing = {os.path.normcase(os.path.abspath(path)) for path in candidates}
        for filename in sorted(os.listdir(sol_dir)):
            if not filename.startswith("sol_step") or not filename.endswith(".pkl"):
                continue
            path = os.path.join(sol_dir, filename)
            key = os.path.normcase(os.path.abspath(path))
            if key not in existing:
                candidates.append(path)
                existing.add(key)
    return candidates


def summarize_solution(label, sol_path, sol, dose_1d, my_plan):
    print("=" * 80)
    print(f"Solution: {label}")
    print(f"Path: {sol_path}")
    print(f"Weight length: {len(sol['optimal_intensity'])}")
    print(f"Dose vector length: {len(dose_1d)}")
    if "act_dose_v" in sol:
        saved_dose = sol["act_dose_v"] * my_plan.get_num_of_fractions()
        max_abs_err = float(np.max(np.abs(saved_dose - dose_1d)))
        mean_abs_err = float(np.mean(np.abs(saved_dose - dose_1d)))
        print(f"Recomputed vs saved dose max abs diff: {max_abs_err:.6f}")
        print(f"Recomputed vs saved dose mean abs diff: {mean_abs_err:.6f}")


def main():
    my_plan, inf_matrix = build_plan(
        data_dir=DATA_DIR,
        patient_id=PATIENT_ID,
        protocol_name=PROTOCOL_NAME,
    )

    sol_paths = get_solution_files(SOLUTION_DIR)
    if not sol_paths:
        raise FileNotFoundError(
            "No solution files found. "
            f"Checked solution dir: {SOLUTION_DIR} and example temp dir: "
            f"{os.path.join(os.path.dirname(__file__), 'Temp')}"
        )

    struct_names = [
        "PTV",
        "ESOPHAGUS",
        "HEART",
        "CORD",
        "LUNG_L",
        "LUNG_R",
        "LUNGS_NOT_GTV",
    ]
    styles = ["-", "--", ":", "-."]
    doses = []
    labels = []

    print(f"Using patient: {PATIENT_ID}")
    print(f"Using data dir: {os.path.abspath(DATA_DIR)}")
    print(f"Using solution dir: {os.path.abspath(SOLUTION_DIR)}")

    for sol_path in sol_paths:
        sol = load_solution(sol_path)
        dose_1d = compute_dose(my_plan, inf_matrix, sol["optimal_intensity"])
        label = os.path.basename(sol_path)
        summarize_solution(label, sol_path, sol, dose_1d, my_plan)
        print("Clinical criteria:")
        pp.Evaluation.display_clinical_criteria(my_plan=my_plan, dose_1d=dose_1d)
        doses.append(dose_1d)
        labels.append(label)

    if PLOT_DVH:
        fig, ax = plt.subplots(figsize=(12, 8))
        for index, dose_1d in enumerate(doses):
            pp.Visualization.plot_dvh(
                my_plan=my_plan,
                dose_1d=dose_1d,
                struct_names=struct_names,
                style=styles[index % len(styles)],
                ax=ax,
            )
        ax.set_title("ECHO-VMAT Solution Comparison DVH")
        ax.legend(labels)
        plt.show()


if __name__ == "__main__":
    main()
