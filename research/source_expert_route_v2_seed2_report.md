# Source-expert route v2 fresh seed=2

Independent private-like mask seed=2; source labels are scoring-only. Baseline is rebuilt by `meta_residual_v2_independent._make_q`; all sensor/dynamic fields are masked on query rows.

## Best alpha per route

 seed             method  alpha    n     rmse
    2   crop_hier_n1_p67   0.50 2644 0.065228
    2  soft_all_r1_l0_p4   0.50 2644 0.065364
    2  soft_all_r2_l0_p4   0.50 2644 0.065423
    2 soft_all_r32_l0_p4   0.25 2644 0.066473
    2          post_mode   0.25 2644 0.066526

## Cohort/source/distance alpha audit

 seed             method          slice    n  alpha_opt  rmse_opt  rmse_a040  rmse_baseline
    2   crop_hier_n1_p67           2025  756   0.554162  0.074530   0.074646       0.076014
    2 soft_crop_r1_l0_p4           2025  756   0.548505  0.074566   0.074673       0.076014
    2  soft_all_r2_l0_p4           2025  756   0.532875  0.074688   0.074771       0.076014
    2  soft_all_r1_l0_p4           2025  756   0.505629  0.074773   0.074828       0.076014
    2          post_mode           2025  756   0.132262  0.075947   0.076219       0.076014
    2 soft_all_r32_l0_p4           2025  756   0.106146  0.075978   0.076250       0.076014
    2   crop_hier_n1_p67            all 2644   0.508993  0.065227   0.065300       0.066805
    2 soft_crop_r1_l0_p4            all 2644   0.514904  0.065298   0.065374       0.066805
    2  soft_all_r1_l0_p4            all 2644   0.503074  0.065364   0.065425       0.066805
    2  soft_all_r2_l0_p4            all 2644   0.499106  0.065423   0.065478       0.066805
    2 soft_all_r32_l0_p4            all 2644   0.274679  0.066470   0.066540       0.066805
    2          post_mode            all 2644   0.237057  0.066525   0.066657       0.066805
    2   crop_hier_n1_p67    far_or_none  494   0.144472  0.066907   0.067206       0.067003
    2  soft_all_r1_l0_p4    far_or_none  494   0.041347  0.066997   0.067427       0.067003
    2 soft_crop_r1_l0_p4    far_or_none  494   0.041347  0.066997   0.067427       0.067003
    2  soft_all_r2_l0_p4    far_or_none  494   0.041347  0.066997   0.067427       0.067003
    2          post_mode    far_or_none  494   0.000000  0.067003   0.067662       0.067003
    2 soft_all_r32_l0_p4    far_or_none  494   0.000000  0.067003   0.067824       0.067003
    2   crop_hier_n1_p67        history 1888   0.493229  0.061100   0.061159       0.062739
    2  soft_all_r1_l0_p4        history 1888   0.502090  0.061192   0.061256       0.062739
    2 soft_crop_r1_l0_p4        history 1888   0.502090  0.061192   0.061256       0.062739
    2  soft_all_r2_l0_p4        history 1888   0.486334  0.061319   0.061364       0.062739
    2 soft_all_r32_l0_p4        history 1888   0.325407  0.062200   0.062228       0.062739
    2          post_mode        history 1888   0.271775  0.062327   0.062419       0.062739
    2          post_mode        mid_2_8  425   0.689201  0.069849   0.070408       0.072971
    2   crop_hier_n1_p67        mid_2_8  425   0.653871  0.070234   0.070653       0.072971
    2 soft_crop_r1_l0_p4        mid_2_8  425   0.680125  0.070251   0.070719       0.072971
    2  soft_all_r1_l0_p4        mid_2_8  425   0.627071  0.070671   0.070977       0.072971
    2  soft_all_r2_l0_p4        mid_2_8  425   0.622608  0.070689   0.070985       0.072971
    2 soft_all_r32_l0_p4        mid_2_8  425   0.623191  0.070709   0.071003       0.072971
    2   crop_hier_n1_p67       near_0_2 1725   0.546653  0.063203   0.063345       0.065138
    2 soft_crop_r1_l0_p4       near_0_2 1725   0.547689  0.063236   0.063376       0.065138
    2  soft_all_r1_l0_p4       near_0_2 1725   0.544392  0.063250   0.063385       0.065138
    2  soft_all_r2_l0_p4       near_0_2 1725   0.541312  0.063343   0.063467       0.065138
    2 soft_all_r32_l0_p4       near_0_2 1725   0.236686  0.064902   0.065014       0.065138
    2          post_mode       near_0_2 1725   0.130842  0.065055   0.065405       0.065138
    2   crop_hier_n1_p67            new 2267   0.533882  0.067608   0.067715       0.069290
    2 soft_crop_r1_l0_p4            new 2267   0.538869  0.067699   0.067805       0.069290
    2  soft_all_r1_l0_p4            new 2267   0.531125  0.067744   0.067840       0.069290
    2  soft_all_r2_l0_p4            new 2267   0.516223  0.067860   0.067933       0.069290
    2 soft_all_r32_l0_p4            new 2267   0.314199  0.068856   0.068888       0.069290
    2          post_mode            new 2267   0.280946  0.068902   0.068972       0.069290
    2   crop_hier_n1_p67       new_2025  379   0.784894  0.093246   0.093779       0.095444
    2 soft_crop_r1_l0_p4       new_2025  379   0.737836  0.093425   0.093852       0.095444
    2  soft_all_r1_l0_p4       new_2025  379   0.688129  0.093689   0.093999       0.095444
    2  soft_all_r2_l0_p4       new_2025  379   0.676598  0.093771   0.094052       0.095444
    2          post_mode       new_2025  379   0.335419  0.095098   0.095111       0.095444
    2 soft_all_r32_l0_p4       new_2025  379   0.237534  0.095307   0.095371       0.095444
    2  soft_all_r2_l0_p4         shared  377   0.393277  0.048148   0.048148       0.049284
    2 soft_crop_r1_l0_p4         shared  377   0.370401  0.048231   0.048238       0.049284
    2   crop_hier_n1_p67         shared  377   0.354377  0.048278   0.048295       0.049284
    2  soft_all_r1_l0_p4         shared  377   0.337030  0.048397   0.048428       0.049284
    2          post_mode         shared  377   0.000000  0.049284   0.050552       0.049284
    2 soft_all_r32_l0_p4         shared  377   0.000000  0.049284   0.050149       0.049284
    2  soft_all_r2_l0_p4    shared_2025  377   0.393277  0.048148   0.048148       0.049284
    2 soft_crop_r1_l0_p4    shared_2025  377   0.370401  0.048231   0.048238       0.049284
    2   crop_hier_n1_p67    shared_2025  377   0.354377  0.048278   0.048295       0.049284
    2  soft_all_r1_l0_p4    shared_2025  377   0.337030  0.048397   0.048428       0.049284
    2          post_mode    shared_2025  377   0.000000  0.049284   0.050552       0.049284
    2 soft_all_r32_l0_p4    shared_2025  377   0.000000  0.049284   0.050149       0.049284
    2   crop_hier_n1_p67 source_landsat 1025   0.554187  0.071509   0.071627       0.073018
    2 soft_crop_r1_l0_p4 source_landsat 1025   0.552985  0.071681   0.071784       0.073018
    2  soft_all_r1_l0_p4 source_landsat 1025   0.551365  0.071683   0.071785       0.073018
    2  soft_all_r2_l0_p4 source_landsat 1025   0.531672  0.071774   0.071851       0.073018
    2 soft_all_r32_l0_p4 source_landsat 1025   0.214515  0.072861   0.072978       0.073018
    2          post_mode source_landsat 1025   0.081869  0.072994   0.073350       0.073018
    2   crop_hier_n1_p67   source_modis  433   0.437736  0.059570   0.059586       0.061663
    2  soft_all_r1_l0_p4   source_modis  433   0.383131  0.060118   0.060121       0.061663
    2 soft_crop_r1_l0_p4   source_modis  433   0.383131  0.060118   0.060121       0.061663
    2  soft_all_r2_l0_p4   source_modis  433   0.368421  0.060359   0.060369       0.061663
    2 soft_all_r32_l0_p4   source_modis  433   0.310359  0.060866   0.060933       0.061663
    2          post_mode   source_modis  433   0.133722  0.061480   0.062201       0.061663
    2 soft_crop_r1_l0_p4      source_s2 1186   0.576890  0.061107   0.061275       0.062874
    2  soft_all_r2_l0_p4      source_s2 1186   0.556998  0.061234   0.061366       0.062874
    2  soft_all_r1_l0_p4      source_s2 1186   0.547854  0.061277   0.061395       0.062874
    2   crop_hier_n1_p67      source_s2 1186   0.518636  0.061370   0.061449       0.062874
    2          post_mode      source_s2 1186   0.438734  0.061985   0.061992       0.062874
    2 soft_all_r32_l0_p4      source_s2 1186   0.299046  0.062505   0.062547       0.062874

No existing candidate was overwritten.
