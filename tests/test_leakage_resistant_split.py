from fod_detection.data.leakage_resistant_split import (
    ImageRecord,
    assign_groups,
    build_sequence_groups,
    verify_split,
)


def record(image_id: str, dhash: int, weather: str = "0", light: str = "0"):
    return ImageRecord(
        image_id=image_id,
        weather=weather,
        light=light,
        labels=("Bolt",),
        object_count=1,
        dhash=dhash,
    )


def test_related_adjacent_frames_stay_in_one_group():
    records = [record("000001", 0), record("000002", 1), record("000003", 3)]

    groups = build_sequence_groups(records, max_dhash_distance=2)

    assert len(groups) == 1
    assert [item.image_id for item in groups[0].records] == [
        "000001",
        "000002",
        "000003",
    ]


def test_large_visual_change_starts_a_new_group():
    records = [record("000001", 0), record("000002", (1 << 20) - 1)]

    groups = build_sequence_groups(records, max_dhash_distance=2)

    assert len(groups) == 2


def test_split_is_complete_unique_and_group_isolated():
    records = [
        record("000001", 0),
        record("000002", 1),
        record("000003", 1, weather="1"),
        record("000004", 0, weather="1"),
        record("000005", 0, light="1"),
        record("000006", 1, light="1"),
    ]
    groups = build_sequence_groups(records, max_dhash_distance=2)
    assignments = assign_groups(
        groups,
        ratios={"train": 0.5, "validation": 0.25, "test": 0.25},
        seed=42,
    )

    # Small synthetic strata may not populate every partition, so add explicit
    # assignments to test the verification rules independently.
    for index, group in enumerate(groups):
        assignments[group.sequence_id] = ("train", "validation", "test")[index % 3]

    result = verify_split(records, groups, assignments)

    assert all(result["checks"].values())
