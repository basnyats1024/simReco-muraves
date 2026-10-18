from __future__ import annotations

from pathlib import Path

import awkward as ak
import numpy as np
import uproot


class RootWriter:

    def __init__(
        self,
        output_prefix="musimData_run4_clustered",
        output_dir="TMP",
        chunk_size=100_000,
    ):

        self.output_prefix = output_prefix
        self.output_dir = Path(output_dir)

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.chunk_size = chunk_size

        self.file_index = 0
        self.n_written = 0

        self._reset_buffers()

    def _reset_buffers(self):

        self.process_id = []
        self.event = []
        self.station = []
        self.cluster_id = []

        self.cluster_multiplicity = []

        self.cluster_energy = []
        self.cluster_position = []
        self.cluster_size = []

        self.cluster_time = []
        self.cluster_time_spread = []

        self.strip_ids = []
        self.strip_energies = []

    def add_record(self, record):

        self.process_id.append(record.process_id)
        self.event.append(record.event)
        self.station.append(record.station)
        self.cluster_id.append(record.cluster_id)

        self.cluster_multiplicity.append(
            record.cluster_multiplicity
        )

        self.cluster_energy.append(
            record.cluster_energy
        )

        self.cluster_position.append(
            record.cluster_position
        )

        self.cluster_size.append(
            record.cluster_size
        )

        self.cluster_time.append(
            record.cluster_time
        )

        self.cluster_time_spread.append(
            record.cluster_time_spread
        )

        self.strip_ids.append(
            record.strip_ids
        )

        self.strip_energies.append(
            record.strip_energies
        )

        if len(self.event) >= self.chunk_size:
            self.flush()

    def extend(self, records):

        for record in records:
            self.add_record(record)

    def _build_tree_data(self):

        return {

            "ProcessID": np.array(
                self.process_id,
                dtype=np.int32,
            ),

            "Event": np.array(
                self.event,
                dtype=np.int32,
            ),

            "Station": np.array(
                self.station,
                dtype=np.int32,
            ),

            "ClusterID": np.array(
                self.cluster_id,
                dtype=np.int32,
            ),

            "ClusterMultiplicity": np.array(
                self.cluster_multiplicity,
                dtype=np.int32,
            ),

            "ClusterEnergy": np.array(
                self.cluster_energy,
                dtype=np.float32,
            ),

            "ClusterPosition": np.array(
                self.cluster_position,
                dtype=np.float32,
            ),

            "ClusterSize": np.array(
                self.cluster_size,
                dtype=np.int32,
            ),

            "ClusterTime": np.array(
                self.cluster_time,
                dtype=np.float32,
            ),

            "ClusterTimeSpread": np.array(
                self.cluster_time_spread,
                dtype=np.float32,
            ),

            "StripIDs": ak.Array(
                self.strip_ids
            ),

            "StripEnergies": ak.Array(
                self.strip_energies
            ),
        }

    def flush(self):

        n_clusters = len(self.event)

        if n_clusters == 0:
            return

        output_file = (
            self.output_dir
            /
            f"{self.output_prefix}_part"
            f"{self.file_index:03d}.root"
        )

        print(
            f"Writing chunk "
            f"{self.file_index:03d} "
            f"({n_clusters} clusters)"
        )

        tree_data = self._build_tree_data()

        with uproot.recreate(
            output_file
        ) as fout:

            fout["ClusteredData"] = tree_data

        self.n_written += n_clusters
        self.file_index += 1

        self._reset_buffers()

    def close(self):

        self.flush()

        print()
        print(
            f"Finished writing "
            f"{self.n_written} clusters"
        )

        print(
            f"Output directory: "
            f"{self.output_dir}"
        )
