"""Build a sequence-aware train, validation and test split for FOD-A."""

from __future__ import annotations

import csv
import json
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class ImageRecord:
    """Metadata needed to group and split one image."""

    image_id: str
    weather: str
    light: str
    labels: tuple[str, ...]
    object_count: int
    dhash: int


@dataclass
class SequenceGroup:
    """A set of consecutive, visually related frames kept together."""

    sequence_id: str
    records: list[ImageRecord]

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def stratum(self) -> tuple[str, str]:
        first = self.records[0]
        return first.weather, first.light


def difference_hash(image_path: Path) -> int:
    """Return a simple 64-bit perceptual difference hash."""

    with Image.open(image_path) as image:
        pixels = list(image.convert("L").resize((9, 8)).getdata())

    bits = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            if left > right:
                bits |= 1 << (row * 8 + column)
    return bits


def hamming_distance(first_hash: int, second_hash: int) -> int:
    """Count different bits between two perceptual hashes."""

    return (first_hash ^ second_hash).bit_count()


def load_environment(csv_path: Path) -> dict[str, tuple[str, str]]:
    """Read weather and lighting codes keyed by image ID."""

    environment = {}
    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            environment[Path(row["File"]).stem] = (row["Weather"], row["Light"])
    return environment


def annotation_labels(xml_path: Path) -> tuple[str, ...]:
    """Read the sorted object labels from one Pascal VOC annotation."""

    root = ET.parse(xml_path).getroot()
    return tuple(sorted(obj.findtext("name", "") for obj in root.findall("object")))


def load_records(voc_root: Path) -> list[ImageRecord]:
    """Load every annotated image in numerical ID order."""

    image_dir = voc_root / "JPEGImages"
    annotation_dir = voc_root / "Annotations"
    environment_path = (
        voc_root
        / "ImageSets"
        / "Main"
        / "CategorizationData"
        / "FOD_categorization_annotations.csv"
    )
    environment = load_environment(environment_path)

    image_ids = sorted(path.stem for path in annotation_dir.glob("*.xml"))
    records = []
    for position, image_id in enumerate(image_ids, start=1):
        image_path = image_dir / f"{image_id}.jpg"
        if image_id not in environment:
            raise ValueError(f"Missing environment metadata for {image_id}")
        if not image_path.exists():
            raise FileNotFoundError(image_path)

        labels = annotation_labels(annotation_dir / f"{image_id}.xml")
        weather, light = environment[image_id]
        records.append(
            ImageRecord(
                image_id=image_id,
                weather=weather,
                light=light,
                labels=labels,
                object_count=len(labels),
                dhash=difference_hash(image_path),
            )
        )
        if position % 5000 == 0:
            print(f"Loaded {position:,} images")

    return records


def frames_are_related(
    previous: ImageRecord,
    current: ImageRecord,
    max_dhash_distance: int,
) -> bool:
    """Decide whether two adjacent IDs belong to the same capture sequence."""

    ids_are_consecutive = int(current.image_id) == int(previous.image_id) + 1
    metadata_matches = (
        current.weather == previous.weather
        and current.light == previous.light
        and current.labels == previous.labels
    )
    visually_similar = hamming_distance(previous.dhash, current.dhash) <= max_dhash_distance
    return ids_are_consecutive and metadata_matches and visually_similar


def build_sequence_groups(
    records: list[ImageRecord],
    max_dhash_distance: int,
) -> list[SequenceGroup]:
    """Create maximal runs of adjacent, related frames."""

    if not records:
        return []

    raw_groups: list[list[ImageRecord]] = [[records[0]]]
    for record in records[1:]:
        if frames_are_related(raw_groups[-1][-1], record, max_dhash_distance):
            raw_groups[-1].append(record)
        else:
            raw_groups.append([record])

    return [
        SequenceGroup(sequence_id=f"sequence_{index:04d}", records=group)
        for index, group in enumerate(raw_groups, start=1)
    ]


def assign_groups(
    groups: list[SequenceGroup],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    """Assign whole groups while balancing each weather-light stratum."""

    if set(ratios) != set(SPLITS) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Ratios must define train, validation and test and sum to one")

    by_stratum: dict[tuple[str, str], list[SequenceGroup]] = defaultdict(list)
    for group in groups:
        by_stratum[group.stratum].append(group)

    assignments = {}
    random_generator = random.Random(seed)

    for stratum_groups in by_stratum.values():
        random_generator.shuffle(stratum_groups)
        stratum_groups.sort(key=lambda group: group.size, reverse=True)

        total_images = sum(group.size for group in stratum_groups)
        targets = {split: total_images * ratios[split] for split in SPLITS}
        counts = Counter({split: 0 for split in SPLITS})

        for group in stratum_groups:
            def assignment_error(candidate: str) -> float:
                error = 0.0
                for split in SPLITS:
                    proposed = counts[split] + (group.size if split == candidate else 0)
                    error += ((proposed - targets[split]) / max(targets[split], 1)) ** 2
                return error

            chosen_split = min(SPLITS, key=assignment_error)
            assignments[group.sequence_id] = chosen_split
            counts[chosen_split] += group.size

    return assignments


def verify_split(
    records: list[ImageRecord],
    groups: list[SequenceGroup],
    assignments: dict[str, str],
) -> dict[str, object]:
    """Check completeness, uniqueness and sequence isolation."""

    all_ids = [record.image_id for record in records]
    grouped_ids = [record.image_id for group in groups for record in group.records]
    group_partitions: dict[str, set[str]] = defaultdict(set)
    partition_ids: dict[str, list[str]] = defaultdict(list)

    for group in groups:
        split = assignments[group.sequence_id]
        group_partitions[group.sequence_id].add(split)
        partition_ids[split].extend(record.image_id for record in group.records)

    duplicate_ids = len(grouped_ids) - len(set(grouped_ids))
    crossing_groups = [
        sequence_id
        for sequence_id, partitions in group_partitions.items()
        if len(partitions) > 1
    ]
    cross_partition_ids = (
        (set(partition_ids["train"]) & set(partition_ids["validation"]))
        | (set(partition_ids["train"]) & set(partition_ids["test"]))
        | (set(partition_ids["validation"]) & set(partition_ids["test"]))
    )

    checks = {
        "all_images_assigned": set(grouped_ids) == set(all_ids),
        "no_duplicate_image_ids": duplicate_ids == 0,
        "no_cross_partition_image_ids": not cross_partition_ids,
        "no_sequence_crosses_partitions": not crossing_groups,
        "all_partitions_non_empty": all(partition_ids[split] for split in SPLITS),
    }
    if not all(checks.values()):
        raise ValueError(f"Split verification failed: {checks}")

    return {
        "checks": checks,
        "image_count": len(all_ids),
        "sequence_count": len(groups),
        "partition_counts": {split: len(partition_ids[split]) for split in SPLITS},
        "duplicate_image_ids": duplicate_ids,
        "cross_partition_image_ids": sorted(cross_partition_ids),
        "crossing_sequence_groups": crossing_groups,
    }


def summarise_conditions(
    groups: list[SequenceGroup],
    assignments: dict[str, str],
) -> dict[str, dict[str, int]]:
    """Count weather-light combinations in each partition."""

    counts: dict[str, Counter[str]] = {split: Counter() for split in SPLITS}
    for group in groups:
        split = assignments[group.sequence_id]
        weather, light = group.stratum
        counts[split][f"weather_{weather}_light_{light}"] += group.size
    return {split: dict(sorted(values.items())) for split, values in counts.items()}


def summarise_sequences(
    records: list[ImageRecord],
    groups: list[SequenceGroup],
    max_dhash_distance: int,
) -> dict[str, object]:
    """Record evidence supporting the sequence-grouping threshold."""

    matching_metadata_distances = []
    for previous, current in zip(records, records[1:]):
        metadata_matches = (
            previous.weather == current.weather
            and previous.light == current.light
            and previous.labels == current.labels
        )
        if metadata_matches:
            matching_metadata_distances.append(
                hamming_distance(previous.dhash, current.dhash)
            )

    retained_pairs = sum(
        distance <= max_dhash_distance
        for distance in matching_metadata_distances
    )
    group_sizes = sorted(group.size for group in groups)
    middle = len(group_sizes) // 2
    median_size = (
        group_sizes[middle]
        if len(group_sizes) % 2
        else (group_sizes[middle - 1] + group_sizes[middle]) / 2
    )

    return {
        "adjacent_pairs_with_matching_metadata": len(matching_metadata_distances),
        "pairs_within_dhash_threshold": retained_pairs,
        "pairs_within_dhash_threshold_percent": round(
            retained_pairs / len(matching_metadata_distances) * 100,
            2,
        ),
        "sequence_size_images": {
            "minimum": min(group_sizes),
            "median": median_size,
            "maximum": max(group_sizes),
        },
    }


def audit_official_partitions(
    voc_root: Path,
    records: list[ImageRecord],
) -> dict[str, object]:
    """Describe the defects repaired by rebuilding the supplied partitions."""

    main_dir = voc_root / "ImageSets" / "Main"
    train_entries = (main_dir / "trainval.txt").read_text(encoding="utf-8").split()
    test_entries = (main_dir / "test.txt").read_text(encoding="utf-8").split()
    train_ids = set(train_entries)
    test_ids = set(test_entries)
    all_ids = {record.image_id for record in records}
    omitted_ids = sorted(all_ids - train_ids - test_ids)

    return {
        "repeated_train_entries_removed": len(train_entries) - len(train_ids),
        "repeated_test_entries_removed": len(test_entries) - len(test_ids),
        "shared_train_test_ids_resolved": sorted(train_ids & test_ids),
        "omitted_images_assigned": len(omitted_ids),
        "omitted_id_range": [omitted_ids[0], omitted_ids[-1]] if omitted_ids else [],
    }


def write_outputs(
    output_dir: Path,
    groups: list[SequenceGroup],
    assignments: dict[str, str],
    summary: dict[str, object],
) -> None:
    """Write the complete CSV manifest, partition ID files and summary."""

    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    partition_ids: dict[str, list[str]] = {split: [] for split in SPLITS}

    for group in groups:
        split = assignments[group.sequence_id]
        for record in group.records:
            partition_ids[split].append(record.image_id)
            rows.append(
                {
                    "image_id": record.image_id,
                    "split": split,
                    "sequence_id": group.sequence_id,
                    "weather": record.weather,
                    "light": record.light,
                    "labels": "|".join(record.labels),
                    "object_count": record.object_count,
                    "dhash": f"{record.dhash:016x}",
                }
            )

    with (output_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    for split, image_ids in partition_ids.items():
        (output_dir / f"{split}.txt").write_text(
            "\n".join(sorted(image_ids)) + "\n",
            encoding="utf-8",
        )

    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
