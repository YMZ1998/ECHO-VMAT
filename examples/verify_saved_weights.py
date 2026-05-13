import argparse
import os
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import portpy.photon as pp

from echo_vmat.arcs import Arcs
from echo_vmat.utils.get_sparse_only import get_sparse_only


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
    return my_plan, inf_matrix, clinical_criteria


def load_weights(sol_path: str = None, npy_path: str = None):
    if sol_path:
        sol_dir = os.path.dirname(sol_path) or "."
        sol_name = os.path.basename(sol_path)
        sol = pp.load_optimal_sol(sol_name=sol_name, path=sol_dir)
        if "optimal_intensity" not in sol:
            raise KeyError(f"'optimal_intensity' not found in solution file: {sol_path}")
        return sol["optimal_intensity"], sol

    if npy_path:
        return np.load(npy_path), None

    raise ValueError("Either sol_path or npy_path must be provided.")


def main():
    parser = argparse.ArgumentParser(
        description="Validate saved ECHO-VMAT weights or solution files in a PortPy/PYLINAC environment."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(".", "data", "data"),
        help="PortPy data directory. Default: ./data/data",
    )
    parser.add_argument(
        "--patient-id",
        default="Lung_Phantom_Patient_1",
        help="Patient id. Default: Lung_Phantom_Patient_1",
    )
    parser.add_argument(
        "--protocol",
        default="Lung_2Gy_30Fx",
        help="Protocol/config prefix. Default: Lung_2Gy_30Fx",
    )
    parser.add_argument(
        "--sol",
        default="./Temp/sol_col_gen",
        help="Path to a saved PortPy solution file, e.g. C:\\Temp\\Lung_Phantom_Patient_1\\sol_step2.pkl",
    )
    parser.add_argument(
        "--weights",
        default=None,
    )
    parser.add_argument(
        "--no-plot",
        default=False,
        help="Skip DVH plotting.",
    )
    args = parser.parse_args()

    if bool(args.sol) == bool(args.weights):
        raise ValueError("Provide exactly one of --sol or --weights.")

    my_plan, inf_matrix, clinical_criteria = build_plan(
        data_dir=args.data_dir,
        patient_id=args.patient_id,
        protocol_name=args.protocol,
    )

    weights, sol = load_weights(sol_path=args.sol, npy_path=args.weights)

    if len(weights) != inf_matrix.A.shape[1]:
        raise ValueError(
            f"Weight length mismatch: got {len(weights)}, expected {inf_matrix.A.shape[1]}. "
            "Check patient, beams/arcs, and influence matrix settings."
        )

    dose_1d = inf_matrix.A @ weights * my_plan.get_num_of_fractions()

    print(f"Loaded weights length: {len(weights)}")
    print(f"Dose vector length: {len(dose_1d)}")
    if sol is not None:
        print(f"Loaded solution file: {args.sol}")
    else:
        print(f"Loaded weight file: {args.weights}")

    pp.Evaluation.display_clinical_criteria(my_plan=my_plan, dose_1d=dose_1d)

    if not args.no_plot:
        struct_names = [
            "PTV",
            "ESOPHAGUS",
            "HEART",
            "CORD",
            "LUNG_L",
            "LUNG_R",
            "LUNGS_NOT_GTV",
        ]
        fig, ax = plt.subplots(figsize=(12, 8))
        pp.Visualization.plot_dvh(
            my_plan=my_plan,
            dose_1d=dose_1d,
            struct_names=struct_names,
            ax=ax,
        )
        ax.set_title("Saved Weight Verification DVH")
        plt.show()


if __name__ == "__main__":
    main()
