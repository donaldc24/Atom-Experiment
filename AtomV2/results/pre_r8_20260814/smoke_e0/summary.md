# smoke_e0 summary

| arm | n | acc_seen_hard | acc_seen_soft | acc_unseen_L1_hard | acc_unseen_L2_hard | acc_unseen_L3_hard | dissociation_gap_hard | soft_hard_gap_seen | closed_map_seen | closed_map_unseen_L1 | closed_map_unseen_L3 | census_atoms_in_use | census_steps_per_token | census_pass_rate | ablation_cv_median | standalone_best_acc_mean_in_use | closed_map_atom_matched_error | closed_map_atom_coverage | decodability_subop_from_delta | decodability_subop_h0_floor | decodability_surface_from_delta | decodability_surface_h0_floor | transfer_task_row_std_mean | transfer_transplant_row_std_mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A0-free | 3 | 0.0010±0.0010 | 0.0003±0.0006 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | -0.0007±0.0006 | 0.5819±0.0184 | 0.5981±0.0304 | 0.6919±0.0258 | 2.6667±1.1547 | 2.7211±0.0866 | 0.1005±0.0299 | -8.3863±0.0690 | 0.0061±0.0060 | 1.2775±0.0040 | 1.0000±0.0000 | 0.6954±0.0174 | 0.8041±0.0047 | 0.4865±0.0052 | 0.6480±0.0364 | 0.0000±0.0000 | 0.0000±0.0000 |
| A0-oracle | 3 | 0.0026±0.0030 | 0.0046±0.0021 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0020±0.0043 | 0.1033±0.0673 | 0.1103±0.0639 | 0.1181±0.0570 | 1.0000±0.0000 | 0.5547±0.4070 | 0.8201±0.1373 | -8.4261±0.0000 | 0.0156±0.0138 | 0.2648±0.0280 | 1.0000±0.0000 | 0.7156±0.0742 | 0.6815±0.0445 | 0.5158±0.1898 | 0.5666±0.0464 | 0.0000±0.0000 | 0.0006±0.0010 |

Composer and atom library are separate line items by rule; never sum them into a 'system size'.
