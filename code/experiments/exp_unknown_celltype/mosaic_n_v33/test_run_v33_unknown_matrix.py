from run_v33_unknown_matrix import expected_unknown_run_keys


def test_unknown_matrix_has_one_teacher_and_three_students_per_target() -> None:
    keys = expected_unknown_run_keys(
        targets=["gdT_2", "NK_3"],
        seeds=[41, 42, 43],
    )
    assert len(keys) == 8
    assert ("gdT_2", 42, "mlp_teacher") in keys
    assert ("gdT_2", 41, "mosaic_full") in keys
    assert ("NK_3", 43, "mosaic_full") in keys

